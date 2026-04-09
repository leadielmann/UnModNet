"""Side-by-side comparisons: origin (LDR) | modulo | reconstructed (tonemapped)."""
import os, argparse, random
import numpy as np
import cv2

def tonemap_reinhard(hdr_float):
    tm = cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)
    # Reinhard expects linear HDR in float32; normalize so the algorithm has headroom.
    x = hdr_float.astype(np.float32)
    if x.max() > 0:
        x = x / x.max() * 8.0  # push into HDR-ish range
    ldr = tm.process(x)
    ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)

def to_uint8_rgb(x):
    x = x.astype(np.float32)
    if x.max() > 255: x = x / x.max() * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)

def main(data_dir, recon_dir, output_dir, n_per_class):
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    made = 0
    for cls in classes:
        modulo_dir = os.path.join(data_dir, cls, "modulo")
        ldr_dir    = os.path.join(data_dir, cls, "ldr")
        if not os.path.isdir(modulo_dir): continue
        names = sorted(f for f in os.listdir(modulo_dir) if f.endswith(".npy"))
        random.Random(0).shuffle(names)
        picked = 0
        for name in names:
            recon_path = os.path.join(recon_dir, name)
            if not os.path.exists(recon_path): continue
            modulo = np.load(os.path.join(modulo_dir, name))
            ldr    = np.load(os.path.join(ldr_dir, name)) if os.path.exists(os.path.join(ldr_dir, name)) else modulo
            recon  = np.load(recon_path)

            panel = np.concatenate([
                to_uint8_rgb(ldr),
                to_uint8_rgb(modulo),
                tonemap_reinhard(recon),
            ], axis=1)
            out = os.path.join(output_dir, f"{cls}_{os.path.splitext(name)[0]}.png")
            cv2.imwrite(out, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            picked += 1; made += 1
            if picked >= n_per_class: break
    print(f"Wrote {made} comparison PNGs to {output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--recon_dir", required=True, help=".../reconstructed_val/unwrapped")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_per_class", type=int, default=3)
    main(**vars(p.parse_args()))