#!/usr/bin/env python3
"""
Generate test data for compress + encrypt pipeline
Compatible with the C++ CUDA code
"""

import numpy as np
import argparse
import os
from pathlib import Path

def generate_random_data(size, data_type='float32'):
    """Generate random floating point data"""
    if data_type == 'normal':
        # Normal distribution
        data = np.random.randn(size).astype(np.float32)
    elif data_type == 'uniform':
        # Uniform distribution [0, 1]
        data = np.random.uniform(0, 1, size).astype(np.float32)
    elif data_type == 'sensor':
        # Simulate sensor data (temperature-like)
        base = 20.0
        noise = np.random.randn(size) * 2.0
        trend = np.sin(np.linspace(0, 4*np.pi, size)) * 5.0
        data = (base + noise + trend).astype(np.float32)
    else:
        # Default: scaled random
        data = (np.random.rand(size) * 100).astype(np.float32)
    
    return data

def generate_image_data(width, height, pattern='gradient'):
    """Generate 2D image data for DCT compression"""
    if pattern == 'gradient':
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        data = (xx + yy) * 127.5
    elif pattern == 'checkerboard':
        data = np.zeros((height, width))
        data[::2, ::2] = 255
        data[1::2, 1::2] = 255
    elif pattern == 'sine':
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        xx, yy = np.meshgrid(x, y)
        data = (np.sin(xx) * np.cos(yy) + 1) * 127.5
    elif pattern == 'random':
        data = np.random.rand(height, width) * 255
    else:
        data = np.ones((height, width)) * 128
    
    return data.astype(np.float32).flatten()

def save_binary(data, filename):
    """Save data as binary file"""
    data.tofile(filename)
    print(f"Saved binary data to {filename}")
    print(f"  Shape: {data.shape}")
    print(f"  Size: {data.nbytes} bytes")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}, Mean: {data.mean():.2f}")

def save_csv(data, filename):
    """Save data as CSV file"""
    np.savetxt(filename, data, delimiter=',', header='value', comments='')
    print(f"Saved CSV data to {filename}")
    print(f"  Size: {len(data)} values")

def main():
    parser = argparse.ArgumentParser(description='Generate test data for compress+encrypt pipeline')
    parser.add_argument('--type', choices=['all', '1d', '2d', 'small', 'large'], 
                        default='all', help='Type of data to generate')
    parser.add_argument('--output-dir', default='data', help='Output directory')
    parser.add_argument('--size', type=int, default=4096, help='Size for 1D data')
    parser.add_argument('--width', type=int, default=512, help='Width for 2D data')
    parser.add_argument('--height', type=int, default=512, help='Height for 2D data')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Generating Test Data for Compress + Encrypt Pipeline")
    print("=" * 60)
    
    if args.type in ['all', '1d', 'small']:
        # Small 1D data (for quick testing)
        print("\n[1] Small 1D Random Data")
        data = generate_random_data(1024, 'uniform')
        save_binary(data, output_dir / 'test_small.bin')
        save_csv(data, output_dir / 'test_small.csv')
    
    if args.type in ['all', '1d']:
        # Medium 1D data
        print("\n[2] Medium 1D Random Data")
        data = generate_random_data(args.size, 'uniform')
        save_binary(data, output_dir / 'test_data.bin')
        save_csv(data, output_dir / 'test_data.csv')
        
        # Sensor-like data
        print("\n[3] Sensor Data")
        data = generate_random_data(args.size, 'sensor')
        save_binary(data, output_dir / 'sensor_data.bin')
        save_csv(data, output_dir / 'sensor_data.csv')
    
    if args.type in ['all', '1d', 'large']:
        # Large 1D data
        print("\n[4] Large 1D Data")
        data = generate_random_data(16384, 'normal')
        save_binary(data, output_dir / 'test_large.bin')
    
    if args.type in ['all', '2d']:
        # 2D image data for DCT
        print("\n[5] 2D Image Data (Gradient)")
        data = generate_image_data(args.width, args.height, 'gradient')
        save_binary(data, output_dir / 'image_gradient.bin')
        
        print("\n[6] 2D Image Data (Sine Pattern)")
        data = generate_image_data(args.width, args.height, 'sine')
        save_binary(data, output_dir / 'image_sine.bin')
        
        print("\n[7] 2D Image Data (Random)")
        data = generate_image_data(args.width, args.height, 'random')
        save_binary(data, output_dir / 'image_random.bin')
    
    if args.type in ['all', '2d']:
        # Square image for DCT (must be multiple of 8)
        sizes = [64, 128, 256, 512]
        for size in sizes:
            print(f"\n[Image {size}x{size}]")
            data = generate_image_data(size, size, 'sine')
            save_binary(data, output_dir / f'image_{size}x{size}.bin')
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)
    
    # Create README
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("Test Data Files for Compress + Encrypt Pipeline\n")
        f.write("=" * 60 + "\n\n")
        f.write("1D Data (for DCB compression):\n")
        f.write("  - test_small.bin/csv: 1024 floats\n")
        f.write("  - test_data.bin/csv: 4096 floats (default)\n")
        f.write("  - sensor_data.bin/csv: 4096 floats (sensor simulation)\n")
        f.write("  - test_large.bin: 16384 floats\n\n")
        f.write("2D Data (for DCT compression):\n")
        f.write("  - image_gradient.bin: gradient pattern\n")
        f.write("  - image_sine.bin: sine wave pattern\n")
        f.write("  - image_random.bin: random noise\n")
        f.write("  - image_*x*.bin: various square sizes\n\n")
        f.write("Usage:\n")
        f.write("  ./compress_encrypt data/test_data.bin dcb\n")
        f.write("  ./compress_encrypt data/image_512x512.bin dct\n")
    
    print(f"\nREADME created: {readme_path}")

if __name__ == '__main__':
    main()