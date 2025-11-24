# UnModNet Fine-tuning on Imagenette

Fork of [original UnModNet](https://github.com/ORIGINAL_AUTHOR/UnModNet) with modifications for Imagenette dataset fine-tuning.

## What's Different

- Multi-class data loaders
- Helper scripts for Imagenette processing
- Full-size (512×512) reconstruction support
- PyTorch 2.6 compatibility fixes

## Setup / Quick Start
```bash
git clone https://github.com/leadielmann/UnModNet.git
cd UnModNet
python -m venv finetuning
source finetuning/bin/activate  # On Windows: finetuning\Scripts\activate

pip install torch torchvision numpy opencv-python Pillow tqdm matplotlib scikit-image
```

## Running the Pipeline

### 1. Convert Images to .npy Format
```bash
python convert_jpeg_to_npy.py \
  --input_dir ~/datasets/imagenette/train \
  --output_dir ../data/converted/train

python convert_jpeg_to_npy.py \
  --input_dir ~/datasets/imagenette/val \
  --output_dir ../data/converted/val
```

### 2. Process Dataset with Modulo Simulation
```bash
python process_all_classes.py \
  --input_dir ../data/converted/train \
  --train_dir ../data/processed/train \
  --test_dir ../data/processed/test \
  --training_sample_per_class 640 \
  --n_cut 5
```

### 3. Generate Edge Maps
```bash
for class in n01440764 n02102040 n02979186 n03000684 n03028079 n03394916 n03417042 n03425413 n03445777 n03888257; do
  python scripts/make_edge_map.py --data_dir ../data/processed/train/$class
done
```

### 4. Train Edge Module
```bash
python execute/train.py -c config/edge_module_finetune.json
```

### 5. Train Mask Module
```bash
python execute/train.py -c config/mask_module_finetune.json
```

### 6. Process Validation Set (Full-Size)
```bash
python process_full_size.py \
  --input_dir ../data/converted/val \
  --output_dir ../data/processed_fullsize/val

for class in n01440764 n02102040 n02979186 n03000684 n03028079 n03394916 n03417042 n03425413 n03445777 n03888257; do
  python scripts/make_edge_map.py --data_dir ../data/processed_fullsize/val/$class
done
```

### 7. Run Reconstruction
```bash
EDGE_MODEL=$(find ../experiments/edge_module -name "model_best.pth")
MASK_MODEL=$(find ../experiments/mask_module -name "model_best.pth")

python execute/infer_LearnMaskNet_fixed.py \
  --resume "$MASK_MODEL" \
  --resume_edge_module "$EDGE_MODEL" \
  --data_dir ../data/processed_fullsize/val \
  --result_dir ../results/reconstructed_fullsize_val \
  default \
  --iter_max 15
```

### 8. Organize Results
```bash
python organize_results.py \
  --data_dir ../data/processed_fullsize/val \
  --result_dir ../results/reconstructed_fullsize_val \
  --output_dir ../results/val_fullsize_by_class
```

Done! Reconstructed images are in `../results/val_fullsize_by_class/`.


# UnModNet: Learning to Unwrap a Modulo Image for High Dynamic Range Imaging

By [Chu Zhou](https://fourson.github.io/), Hang Zhao, Jin Han, Chang Xu, Chao Xu, Tiejun Huang, [Boxin Shi](http://ci.idm.pku.edu.cn/)
![Network](Network.png)

[PDF](https://proceedings.neurips.cc/paper/2020/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf) | [SUPP](https://proceedings.neurips.cc/paper/2020/file/1102a326d5f7c9e04fc3c89d0ede88c9-Supplemental.pdf)

## Abstract
A conventional camera often suffers from over- or under-exposure when recording a real-world scene with a very high dynamic range (HDR). In contrast, a modulo camera with a Markov random field (MRF) based unwrapping algorithm can theoretically accomplish unbounded dynamic range but shows degenerate performances when there are modulus-intensity ambiguity, strong local contrast, and color misalignment. In this paper, we reformulate the modulo image unwrapping problem into a series of binary labeling problems and propose a modulo edge-aware model, named as UnModNet, to iteratively estimate the binary rollover masks of the modulo image for unwrapping. Experimental results show that our approach can generate 12-bit HDR images from 8-bit modulo images reliably, and runs much faster than the previous MRF-based algorithm thanks to the GPU acceleration.
## Prerequisites

* Linux Distributions (tested on Ubuntu 18.04).
* NVIDIA GPU and CUDA cuDNN
* Python >= 3.7
* Pytorch >= 1.1.0
* cv2
* numpy
* tqdm
* tensorboardX (for training visualization)

## Inference

* To unwrap RGB modulo images (in `.npy` format and in `(H, W, 3)` shape):
```
python execute/infer_LearnMaskNet.py -r checkpoint/checkpoint-mask.pth --data_dir <path_to_modulo_images> --result_dir <path_to_result> --resume_edge_module checkpoint/checkpoint-edge.pth default
```

* To unwrap grayscale modulo images (in `.npy` format and in `(H, W, 1)` shape):
```
python execute/infer_LearnMaskNet.py -r checkpoint/checkpoint-mask-gray.pth --data_dir <path_to_modulo_images> --result_dir <path_to_result> --resume_edge_module checkpoint/checkpoint-edge-gray.pth default
```

* Use `TonemapReinhard_npy.py` to visualize the results. Note that the default tonemap method we use is `cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)`.

## Pre-trained models and test examples

https://drive.google.com/drive/folders/10Y8MOr2o2TZzTI5RZUQZQ-0RBezbzhIV?usp=sharing

## Training your own model

1. Make dataset from original data (HDR images in `.npy` format):
    * make dataset:
    ```
    python scripts/make_dataset.py --data_dir <path_to_original_data> --train_dir <path_to_training_dataset> --test_dir <path_to_test_dataset> --training_sample <number_of_training_samples>
    ```
    * make edge map:
    ```
    python scripts/make_edge_map.py --data_dir <path_to_training_dataset>
    ```

2. Configure the training parameters:
    * write your own `config.json` or use ours: `config/edge_module.json` and `config/mask_module.json` for two stages respectively
    * edit the learning rate schedule function (LambdaLR) at `get_lr_lambda` in `utils/util.py`

3. Run:
```
    python execute/train.py -c <path_to_config_file>
```

## Citation

If you find this work helpful to your research, please cite:
```
@inproceedings{NEURIPS2020_1102a326,
 author = {Zhou, Chu and Zhao, Hang and Han, Jin and Xu, Chang and Xu, Chao and Huang, Tiejun and Shi, Boxin},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {1559--1570},
 publisher = {Curran Associates, Inc.},
 title = {UnModNet: Learning to Unwrap a Modulo Image for High Dynamic Range Imaging},
 url = {https://proceedings.neurips.cc/paper/2020/file/1102a326d5f7c9e04fc3c89d0ede88c9-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
