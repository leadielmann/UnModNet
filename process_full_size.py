import os
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')

# Import the dataset creation functions
from scripts.make_dataset import RandomExposer, DataItem

def process_full_size(input_dir, output_dir, modulo_bits=8, ref_bits=12):
    """
    Process images at full resolution (512×512) without cropping
    """
    modulo_pixel_max = 2 ** modulo_bits - 1
    ref_pixel_max = 2 ** ref_bits - 1
    
    # Get all class directories
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"Processing {len(classes)} classes at full resolution (512×512)")
    
    for class_name in classes:
        print(f"\nProcessing class: {class_name}")
        
        class_input = os.path.join(input_dir, class_name)
        class_output = os.path.join(output_dir, class_name)
        
        # Create output subdirectories
        subdirs = ['modulo', 'origin', 'fold_number', 'mask', 'ref', 'ldr', 
                   'modulo_edge_dir', 'fold_number_edge']
        for subdir in subdirs:
            os.makedirs(os.path.join(class_output, subdir), exist_ok=True)
        
        # Get all .npy files
        files = sorted([f for f in os.listdir(class_input) if f.endswith('.npy')])
        
        for filename in tqdm(files, desc=f"  {class_name}", ascii=True):
            # Load full-size image (512×512)
            img = np.load(os.path.join(class_input, filename))
            
            # Apply random exposure
            random_exposer = RandomExposer(img)
            random_exposer.expose(modulo_pixel_max, ref_pixel_max)
            
            if random_exposer.success:
                # Create modulo data
                data_item = DataItem(random_exposer.hdr, modulo_pixel_max)
                
                # Save all components
                np.save(os.path.join(class_output, 'modulo', filename), data_item.modulo)
                np.save(os.path.join(class_output, 'origin', filename), data_item.origin)
                np.save(os.path.join(class_output, 'fold_number', filename), data_item.fold_number)
                np.save(os.path.join(class_output, 'mask', filename), data_item.mask)
                np.save(os.path.join(class_output, 'ref', filename), data_item.ref)
                np.save(os.path.join(class_output, 'ldr', filename), data_item.ldr)
        
        print(f"Processed {len(files)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images at full resolution')
    parser.add_argument('--input_dir', required=True, help='Converted images directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    process_full_size(args.input_dir, args.output_dir)