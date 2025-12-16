import os
import numpy as np
from PIL import Image
import argparse

def create_dummy_data(count=100, output_dir='data/celeba_preprocessed', size=64):
    """
    Creates dummy random images to test the training pipeline.
    Useful for Colab/Smoke tests when real data isn't uploaded yet.
    """
    print(f"Creating {count} dummy images in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(count):
        # Generate random noise image
        img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save
        filename = f"dummy_{i:04d}.jpg"
        path = os.path.join(output_dir, filename)
        img.save(path)
        
    print(f"✅ Created {count} images. You can now run 'python train.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='data/celeba_preprocessed', help='Output directory for dummy images')
    parser.add_argument('--size', type=int, default=64, help='Image size (creates size x size images)')
    args = parser.parse_args()
    
    create_dummy_data(count=args.count, output_dir=args.output_dir, size=args.size)
