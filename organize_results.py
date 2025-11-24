import os
import shutil
import numpy as np
from pathlib import Path

def organize_by_class(data_dir, result_dir, output_dir):
    """
    Organize reconstructed results back into class folders
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create output class directories
    for class_name in classes:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # Map filenames to classes
    file_to_class = {}
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name, 'modulo')
        if os.path.exists(class_path):
            files = [f.replace('.npy', '') for f in os.listdir(class_path) if f.endswith('.npy')]
            for f in files:
                file_to_class[f] = class_name
    
    print(f"Mapped {len(file_to_class)} filenames to classes")
    
    # Move reconstructed files to appropriate class folders
    unwrapped_dir = os.path.join(result_dir, 'unwrapped')
    
    if not os.path.exists(unwrapped_dir):
        print(f"Unwrapped directory not found: {unwrapped_dir}")
        return
    
    moved_count = 0
    not_found_count = 0
    
    for filename in os.listdir(unwrapped_dir):
        if filename.endswith('.npy'):
            file_base = filename.replace('.npy', '')
            
            if file_base in file_to_class:
                class_name = file_to_class[file_base]
                src = os.path.join(unwrapped_dir, filename)
                dst = os.path.join(output_dir, class_name, filename)
                
                shutil.copy2(src, dst)
                moved_count += 1
                
                if moved_count % 100 == 0:
                    print(f"  Processed {moved_count} files...")
            else:
                not_found_count += 1
                if not_found_count <= 5:  # Only print first 5
                    print(f"No class found for: {filename}")
    
    print(f"\n Organized {moved_count} files into {len(classes)} class folders")
    
    if not_found_count > 0:
        print(f"⚠️  {not_found_count} files could not be mapped to classes")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary by class:")
    print("="*60)
    total = 0
    for class_name in classes:
        class_path = os.path.join(output_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.npy')])
            print(f"  {class_name}: {count} files")
            total += count
    print("="*60)
    print(f"Total: {total} files")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize reconstruction results by class')
    parser.add_argument('--data_dir', required=True, help='Original data directory (e.g., ../data/processed/val)')
    parser.add_argument('--result_dir', required=True, help='Reconstruction results directory (e.g., ../results/reconstructed_val)')
    parser.add_argument('--output_dir', required=True, help='Output directory for organized results (e.g., ../results/val_by_class)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Organizing Reconstruction Results")
    print("="*60)
    print(f"Data dir: {args.data_dir}")
    print(f"Result dir: {args.result_dir}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    print()
    
    organize_by_class(args.data_dir, args.result_dir, args.output_dir)