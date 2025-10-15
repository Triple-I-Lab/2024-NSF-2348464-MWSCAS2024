#!/usr/bin/env python3
import random
import string
import numpy as np
import os
from PIL import Image

def generate_random_text(num_chars, output_file):
    """Generate random text data"""
    print(f"Generating {num_chars} characters of random text...")
    
    # Mix of random words and repeated patterns for better compression
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
             for _ in range(1000)]
    
    text = []
    total_chars = 0
    
    while total_chars < num_chars:
        word = random.choice(words)
        text.append(word)
        text.append(' ')
        total_chars += len(word) + 1
    
    final_text = ''.join(text)[:num_chars]
    
    with open(output_file, 'w') as f:
        f.write(final_text)
    
    print(f"Saved to {output_file} ({len(final_text)} bytes)")
    return output_file

def generate_random_numbers(num_entries, output_file):
    """Generate random numbers for compression testing"""
    print(f"Generating {num_entries} random numbers...")
    
    data = [random.randint(1, 1000) for _ in range(num_entries)]
    
    with open(output_file, 'w') as f:
        for num in data:
            f.write(f"{num}\n")
    
    file_size = os.path.getsize(output_file)
    print(f"Saved to {output_file} ({file_size} bytes)")
    return output_file

def generate_random_image(width, height, output_file):
    """Generate random image data"""
    print(f"Generating {width}x{height} random image...")
    
    # Create random RGB image
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add some structure for better compression
    # Add gradient
    gradient = np.linspace(0, 255, width).astype(np.uint8)
    data[:, :, 0] = (data[:, :, 0] * 0.5 + gradient * 0.5).astype(np.uint8)
    
    # Add some repeated blocks
    block_size = 32
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if random.random() > 0.5:
                color = [random.randint(0, 255) for _ in range(3)]
                data[i:i+block_size, j:j+block_size] = color
    
    img = Image.fromarray(data, 'RGB')
    img.save(output_file)
    
    file_size = os.path.getsize(output_file)
    print(f"Saved to {output_file} ({file_size} bytes)")
    return output_file

def generate_test_dataset(base_dir='data'):
    """Generate complete test dataset"""
    os.makedirs(base_dir, exist_ok=True)
    
    print("="*60)
    print("Generating Test Dataset")
    print("="*60)
    
    files = {}
    
    # Text files of various sizes
    text_sizes = [1000, 5000, 10000, 50000, 100000]
    for size in text_sizes:
        filename = f"{base_dir}/text_{size}.txt"
        generate_random_text(size, filename)
        files[f'text_{size}'] = filename
    
    # Number files
    number_sizes = [100, 500, 1000, 5000, 10000]
    for size in number_sizes:
        filename = f"{base_dir}/numbers_{size}.txt"
        generate_random_numbers(size, filename)
        files[f'numbers_{size}'] = filename
    
    # Images of various sizes
    image_sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    for width, height in image_sizes:
        filename = f"{base_dir}/image_{width}x{height}.png"
        generate_random_image(width, height, filename)
        files[f'image_{width}x{height}'] = filename
    
    print("\n" + "="*60)
    print(f"Generated {len(files)} test files in {base_dir}/")
    print("="*60)
    
    # Save file list
    with open(f"{base_dir}/file_list.txt", 'w') as f:
        for name, path in files.items():
            size = os.path.getsize(path)
            f.write(f"{name}: {path} ({size} bytes)\n")
    
    return files

def generate_paper_dataset():
    """Generate datasets matching the paper's experiments"""
    print("\n" + "="*60)
    print("Generating Paper-Specific Datasets")
    print("="*60)
    
    os.makedirs('data/paper', exist_ok=True)
    
    # Text data sizes from paper (Table I)
    text_sizes = [10000, 20000, 40000, 60000, 80000]
    
    for size in text_sizes:
        filename = f"data/paper/text_{size//1000}k.txt"
        generate_random_text(size, filename)
    
    # Image data sizes from paper (Figure 2)
    image_sizes = [
        (64, 64),    # ~5k pixels
        (128, 128),  # ~10k pixels
        (181, 181),  # ~15k pixels
        (256, 256),  # ~20k pixels
    ]
    
    for width, height in image_sizes:
        filename = f"data/paper/image_{width}x{height}.png"
        generate_random_image(width, height, filename)
    
    print(f"\nPaper datasets saved in data/paper/")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test data for compression experiments')
    parser.add_argument('--type', choices=['text', 'image', 'numbers', 'all', 'paper'], 
                       default='all', help='Type of data to generate')
    parser.add_argument('--size', type=int, help='Size of data (chars for text, pixels for image)')
    parser.add_argument('--output', default='data/test', help='Output file/directory')
    
    args = parser.parse_args()
    
    if args.type == 'all':
        generate_test_dataset()
        generate_paper_dataset()
    elif args.type == 'paper':
        generate_paper_dataset()
    elif args.type == 'text':
        size = args.size or 10000
        generate_random_text(size, args.output + '.txt')
    elif args.type == 'image':
        size = args.size or 256
        generate_random_image(size, size, args.output + '.png')
    elif args.type == 'numbers':
        size = args.size or 1000
        generate_random_numbers(size, args.output + '.txt')