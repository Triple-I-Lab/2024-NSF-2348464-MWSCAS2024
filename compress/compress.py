#!/usr/bin/env python3
import numpy as np
import zlib
import sys
from PIL import Image

def compress_text(input_file, output_file):
    """Compress text/sensor data using float16 + zlib"""
    data = np.fromfile(input_file, dtype=np.float32)
    print(f"[TEXT] Original: {len(data)} floats, {data.nbytes} bytes")
    
    # Convert float32 -> float16 (half precision, 50% size reduction)
    data_fp16 = data.astype(np.float16)
    print(f"[TEXT] Float16: {data_fp16.nbytes} bytes")
    
    # Compress with zlib (better than gzip for small data)
    compressed = zlib.compress(data_fp16.tobytes(), level=9)
    print(f"[TEXT] Compressed: {len(compressed)} bytes")
    print(f"[TEXT] Ratio: {(1 - len(compressed)/data.nbytes)*100:.1f}%")
    
    np.frombuffer(compressed, dtype=np.uint8).tofile(output_file)
    
    # Save metadata
    meta_file = output_file.replace('.bin', '_meta.txt')
    with open(meta_file, 'w') as f:
        f.write(f"method=text\n")
        f.write(f"original_size={len(data)}\n")
        f.write(f"dtype=float16\n")
    
    print(f"[TEXT] Saved: {output_file}")
    print(f"[TEXT] Metadata: {meta_file}")

def compress_image(input_file, output_file):
    """Compress image data using JPEG-style quantization + zlib"""
    data = np.fromfile(input_file, dtype=np.float32)
    
    # Reshape to square image
    size = int(np.sqrt(len(data)))
    if size * size != len(data):
        size = int(np.ceil(np.sqrt(len(data))))
        data = np.pad(data, (0, size*size - len(data)), mode='constant')
    
    image = data.reshape(size, size)
    print(f"[IMAGE] Original: {size}x{size}, {data.nbytes} bytes")
    
    # Normalize to 0-255 and convert to uint8
    min_val, max_val = image.min(), image.max()
    normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Save as PNG
    img = Image.fromarray(normalized, mode='L')
    temp_file = output_file.replace('.bin', '.png')
    img.save(temp_file, 'PNG', optimize=True)
    
    # Compress PNG with zlib
    with open(temp_file, 'rb') as f:
        png_data = f.read()
    compressed = zlib.compress(png_data, level=9)
    
    print(f"[IMAGE] PNG size: {len(png_data)} bytes")
    print(f"[IMAGE] Compressed: {len(compressed)} bytes")
    print(f"[IMAGE] Ratio: {(1 - len(compressed)/data.nbytes)*100:.1f}%")
    
    np.frombuffer(compressed, dtype=np.uint8).tofile(output_file)
    
    # Save metadata
    meta_file = output_file.replace('.bin', '_meta.txt')
    with open(meta_file, 'w') as f:
        f.write(f"method=image\n")
        f.write(f"original_size={len(data)}\n")
        f.write(f"width={size}\n")
        f.write(f"height={size}\n")
        f.write(f"min={min_val}\n")
        f.write(f"max={max_val}\n")
        f.write(f"dtype=float32\n")
    
    print(f"[IMAGE] Saved: {output_file}")
    print(f"[IMAGE] Metadata: {meta_file}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 compress.py <input.bin> <method>")
        print("  method: text or image")
        print("\nExamples:")
        print("  python3 compress.py data/sensor_data.bin text")
        print("  python3 compress.py data/image_512x512.bin image")
        sys.exit(1)
    
    input_file = sys.argv[1]
    method = sys.argv[2].lower()
    output_file = input_file.replace('.bin', '_compressed.bin')
    
    if method == 'text':
        compress_text(input_file, output_file)
    elif method == 'image':
        compress_image(input_file, output_file)
    else:
        print(f"Error: Unknown method '{method}'. Use 'text' or 'image'")
        sys.exit(1)