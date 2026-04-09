#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG — edit these ======
IMAGENETTE_ROOT="${HOME}/datasets/imagenette2"   # contains train/ and val/
DATA_ROOT="../data"                              # sibling of repo
EXP_ROOT="../experiments"
PRETRAINED_DIR="../pretrained_models"            # checkpoint-edge.pth, checkpoint-mask.pth
RESULTS_ROOT="../results"
TRAIN_SAMPLES_PER_CLASS=640
N_CUT=5
CLASSES="n01440764 n02102040 n02979186 n03000684 n03028079 n03394916 n03417042 n03425413 n03445777 n03888257"
# =================================

stage="${1:-all}"
log() { echo -e "\n\033[1;34m[$(date +%H:%M:%S)] $*\033[0m"; }

run_convert() {
  log "Stage 1/7: JPEG -> .npy"
  for split in train val; do
    out="${DATA_ROOT}/converted/${split}"
    if [[ -d "$out" ]] && [[ -n "$(ls -A "$out" 2>/dev/null)" ]]; then
      log "  skip ${split} (exists)"; continue
    fi
    python convert_jpeg_to_npy.py \
      --input_dir  "${IMAGENETTE_ROOT}/${split}" \
      --output_dir "$out"
  done
}

run_process_train() {
  log "Stage 2/7: modulo simulation (train, 256x256 crops)"
  out_train="${DATA_ROOT}/processed/train"
  if [[ -d "$out_train" ]] && [[ -n "$(ls -A "$out_train" 2>/dev/null)" ]]; then
    log "  skip (exists)"; return
  fi
  python process_all_classes.py \
    --input_dir "${DATA_ROOT}/converted/train" \
    --train_dir "${DATA_ROOT}/processed/train" \
    --test_dir  "${DATA_ROOT}/processed/test" \
    --training_sample_per_class "${TRAIN_SAMPLES_PER_CLASS}" \
    --n_cut "${N_CUT}"
}

run_edge_maps_train() {
  log "Stage 3/7: edge maps for training set"
  for cls in $CLASSES; do
    dir="${DATA_ROOT}/processed/train/${cls}"
    [[ -d "${dir}/modulo_edge_dir" ]] && { log "  skip $cls"; continue; }
    python scripts/make_edge_map.py --data_dir "$dir"
  done
}

run_train_edge() {
  log "Stage 4/7: fine-tune edge module"
  # CRITICAL: -r loads pretrained weights; without it you train from scratch.
  python execute/train.py \
    -c config/edge_module_finetune.json \
    -r "${PRETRAINED_DIR}/checkpoint-edge.pth"
}

run_train_mask() {
  log "Stage 5/7: fine-tune mask module"
  python execute/train.py \
    -c config/mask_module_finetune.json \
    -r "${PRETRAINED_DIR}/checkpoint-mask.pth"
}

run_process_val() {
  log "Stage 6a/7: modulo simulation (val, full 512x512)"
  out="${DATA_ROOT}/processed_fullsize/val"
  if [[ ! -d "$out" ]] || [[ -z "$(ls -A "$out" 2>/dev/null)" ]]; then
    python process_full_size.py \
      --input_dir  "${DATA_ROOT}/converted/val" \
      --output_dir "$out"
  fi
  log "Stage 6b/7: edge maps for val set"
  for cls in $CLASSES; do
    dir="${out}/${cls}"
    [[ -d "${dir}/modulo_edge_dir" ]] && continue
    python scripts/make_edge_map.py --data_dir "$dir"
  done
}

run_infer() {
  log "Stage 7a/7: inference on val set"
  EDGE_CKPT="$(find "${EXP_ROOT}/edge_module" -name 'model_best.pth' | head -n1)"
  MASK_CKPT="$(find "${EXP_ROOT}/mask_module" -name 'model_best.pth' | head -n1)"
  [[ -z "$EDGE_CKPT" || -z "$MASK_CKPT" ]] && { echo "ERROR: no model_best.pth found"; exit 1; }
  log "  using edge: $EDGE_CKPT"
  log "  using mask: $MASK_CKPT"
  python execute/infer_LearnMaskNet_fixed.py \
    -r "$MASK_CKPT" \
    --resume_edge_module "$EDGE_CKPT" \
    --data_dir   "${DATA_ROOT}/processed_fullsize/val" \
    --result_dir "${RESULTS_ROOT}/reconstructed_val" \
    default --iter_max 15

  log "Stage 7b/7: organize by class"
  python organize_results.py \
    --data_dir   "${DATA_ROOT}/processed_fullsize/val" \
    --result_dir "${RESULTS_ROOT}/reconstructed_val" \
    --output_dir "${RESULTS_ROOT}/val_by_class"

  log "Stage 7c/7: visualizations"
  python scripts/visualize_results.py \
    --data_dir    "${DATA_ROOT}/processed_fullsize/val" \
    --recon_dir   "${RESULTS_ROOT}/reconstructed_val/unwrapped" \
    --output_dir  "${RESULTS_ROOT}/visualizations" \
    --n_per_class 3
}

case "$stage" in
  convert)      run_convert ;;
  process)      run_process_train ;;
  edge_maps)    run_edge_maps_train ;;
  train_edge)   run_train_edge ;;
  train_mask)   run_train_mask ;;
  process_val)  run_process_val ;;
  infer)        run_infer ;;
  all)          run_convert; run_process_train; run_edge_maps_train; \
                run_train_edge; run_train_mask; run_process_val; run_infer ;;
  *) echo "Usage: $0 {convert|process|edge_maps|train_edge|train_mask|process_val|infer|all}"; exit 1 ;;
esac

log "Done: stage=$stage"