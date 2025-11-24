import os
import argparse
from pathlib import Path
import subprocess

def process_all_classes(input_dir, train_dir, test_dir, training_sample_per_class, n_cut):
    """Process all class folders while preserving structure"""
    
    input_path = Path(input_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)
    
    # Get all class directories
    class_dirs = sorted([d.name for d in input_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(class_dirs)} classes: {class_dirs}")
    print(f"Processing with {training_sample_per_class} training samples per class...")
    print("="*60)
    
    for class_name in class_dirs:
        print(f"\n Processing class: {class_name}")
        
        class_input = input_path / class_name
        class_train = train_path / class_name
        class_test = test_path / class_name
        
        # Create output directories
        class_train.mkdir(parents=True, exist_ok=True)
        class_test.mkdir(parents=True, exist_ok=True)
        
        # Run make_dataset.py for this class
        cmd = [
            "python", "scripts/make_dataset.py",
            "--data_dir", str(class_input),
            "--train_dir", str(class_train),
            "--test_dir", str(class_test),
            "--training_sample", str(training_sample_per_class),
            "--n_cut", str(n_cut)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Success: {class_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {class_name}")
            print(f"{e.stderr}")
    
    print("\n" + "="*60)
    print("All classes processed!")
    
    # Summary
    print("\nSummary:")
    for class_name in class_dirs:
        train_files = len(list((train_path / class_name / "modulo").glob("*.npy")))
        test_files = len(list((test_path / class_name / "modulo").glob("*.npy")))
        print(f"  {class_name}: {train_files} train, {test_files} test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process all classes while preserving folder structure')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with class folders')
    parser.add_argument('--train_dir', type=str, required=True, help='Output training directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Output test directory')
    parser.add_argument('--training_sample_per_class', type=int, default=80, 
                        help='Number of training samples per class')
    parser.add_argument('--n_cut', type=int, default=5, help='Number of crops per image')
    
    args = parser.parse_args()
    
    process_all_classes(
        args.input_dir,
        args.train_dir,
        args.test_dir,
        args.training_sample_per_class,
        args.n_cut
    )