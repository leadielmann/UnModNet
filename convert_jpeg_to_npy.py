import numpy as np
from PIL import Image
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def convert_images_to_npy(input_dir, output_dir, target_size=512):
    """Convert JPEG images to .npy format, resizing to target_size×target_size"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    total_converted = 0
    
    for class_dir in class_dirs:
        # Create output class directory
        output_class_dir = output_path / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in this class
        images = list(class_dir.glob('*.JPEG')) + list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        print(f"\nConverting {len(images)} images in class: {class_dir.name}")
        
        for img_path in tqdm(images, desc=f"  {class_dir.name}", ascii=True):
            try:
                # Load image
                img = Image.open(img_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target_size × target_size
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # Verify shape
                assert img_array.shape == (target_size, target_size, 3), f"Unexpected shape: {img_array.shape}"
                
                # Save as .npy
                output_file = output_class_dir / (img_path.stem + '.npy')
                np.save(output_file, img_array)
                
                total_converted += 1
                
            except Exception as e:
                print(f"\n Error converting {img_path.name}: {e}")
        
        print(f"Completed {class_dir.name}: {len(list(output_class_dir.glob('*.npy')))} files")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete! Total: {total_converted} images")
    print(f"All images resized to {target_size}×{target_size}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JPEG images to .npy format for UnModNet')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory with class folders containing JPEG images')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for .npy files')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Target size for resizing (default: 512)')
    
    args = parser.parse_args()
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target size: {args.target_size}×{args.target_size}")
    
    convert_images_to_npy(args.input_dir, args.output_dir, args.target_size)