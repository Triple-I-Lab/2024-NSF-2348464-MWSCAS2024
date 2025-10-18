#!/usr/bin/env python3
import numpy as np
import zlib
import sys
from PIL import Image

def read_metadata(meta_file):
    """Read metadata file"""
    meta = {}
    with open(meta_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            meta[key] = value
    return meta

def decompress_text(input_file, output_file, meta):
    """Decompress text/sensor data"""
    encrypted = np.fromfile(input_file, dtype=np.uint8)
    print(f"[TEXT] Loaded: {len(encrypted)} bytes")
    
    try:
        # Decompress zlib
        decompressed = zlib.decompress(encrypted.tobytes())
        
        # Load as float16 then convert to float32
        data_fp16 = np.frombuffer(decompressed, dtype=np.float16)
        data = data_fp16.astype(np.float32)
        print(f"[TEXT] Decompressed: {len(data)} floats")
        
        data.tofile(output_file)
        print(f"[TEXT] Saved: {output_file}")
        
        print(f"\n[TEXT] Stats:")
        print(f"  First 10: {data[:10]}")
        print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}, Mean: {data.mean():.2f}")
        
    except Exception as e:
        print(f"[TEXT] Error: {e}")

def decompress_image(input_file, output_file, meta):
    """Decompress image data"""
    encrypted = np.fromfile(input_file, dtype=np.uint8)
    print(f"[IMAGE] Loaded: {len(encrypted)} bytes")
    
    try:
        # Decompress zlib
        decompressed = zlib.decompress(encrypted.tobytes())
        
        # Save as temporary PNG
        temp_file = output_file.replace('.bin', '_temp.png')
        with open(temp_file, 'wb') as f:
            f.write(decompressed)
        
        # Load PNG
        img = Image.open(temp_file)
        normalized = np.array(img, dtype=np.uint8)
        
        print(f"[IMAGE] Image size: {normalized.shape}")
        
        # Denormalize back to original range
        min_val = float(meta.get('min', 0))
        max_val = float(meta.get('max', 255))
        data = (normalized.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
        
        # Flatten and remove padding if needed
        data_flat = data.flatten()
        original_size = int(meta.get('original_size', len(data_flat)))
        data_flat = data_flat[:original_size]
        
        data_flat.tofile(output_file)
        print(f"[IMAGE] Saved: {output_file}")
        
        print(f"\n[IMAGE] Stats:")
        print(f"  Shape: {data.shape}")
        print(f"  Min: {data_flat.min():.2f}, Max: {data_flat.max():.2f}, Mean: {data_flat.mean():.2f}")
        
        # Clean up temp file
        import os
        os.remove(temp_file)
        
    except Exception as e:
        print(f"[IMAGE] Error: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 decompress.py <encrypted.bin>")
        print("\nExample:")
        print("  python3 decompress.py data/sensor_data_encrypted.bin")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('_encrypted', '_decompressed')
    
    # Read metadata
    meta_file = input_file.replace('_encrypted.bin', '_compressed_meta.txt')
    try:
        meta = read_metadata(meta_file)
        method = meta.get('method', 'text')
        print(f"Method: {method}")
    except:
        print("Warning: Metadata not found, assuming text method")
        method = 'text'
        meta = {}
    
    if method == 'text':
        decompress_text(input_file, output_file, meta)
    elif method == 'image':
        decompress_image(input_file, output_file, meta)
    else:
        print(f"Error: Unknown method '{method}'")
        sys.exit(1)