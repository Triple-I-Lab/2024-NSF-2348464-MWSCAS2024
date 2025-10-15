import cv2
import numpy as np
import sys
import time
import pickle

def dct_compress(image, quality=50):
    """DCT-based image compression"""
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Split channels
    y, cr, cb = cv2.split(ycrcb)
    
    # Compress Y channel with DCT
    compressed_y = []
    h, w = y.shape
    
    # Process 8x8 blocks
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = y[i:i+8, j:j+8].astype(np.float32)
            if block.shape == (8, 8):
                # DCT transform
                dct_block = cv2.dct(block)
                # Quantization
                quant_block = np.round(dct_block / quality)
                compressed_y.append(quant_block)
    
    return compressed_y, (h, w), cr, cb

def dct_decompress(compressed_y, shape, cr, cb, quality=50):
    """DCT-based image decompression"""
    h, w = shape
    y_reconstructed = np.zeros((h, w), dtype=np.float32)
    
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if i+8 <= h and j+8 <= w:
                # Dequantization
                dequant_block = compressed_y[idx] * quality
                # Inverse DCT
                idct_block = cv2.idct(dequant_block)
                y_reconstructed[i:i+8, j:j+8] = idct_block
                idx += 1
    
    # Reconstruct image
    ycrcb = cv2.merge([y_reconstructed.astype(np.uint8), cr, cb])
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return image

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/test.jpg'
    
    # Read image
    image = cv2.imread(input_file)
    if image is None:
        print(f"Cannot read {input_file}")
        sys.exit(1)
    
    # Resize to standard size for testing
    image = cv2.resize(image, (512, 512))
    
    original_size = image.nbytes
    print(f"Original size: {original_size} bytes")
    
    # Compress
    start = time.time()
    compressed, shape, cr, cb = dct_compress(image, quality=50)
    compress_time = (time.time() - start) * 1000
    
    # Save compressed data
    with open('compressed_image.pkl', 'wb') as f:
        pickle.dump((compressed, shape, cr, cb), f)
    
    compressed_size = len(pickle.dumps((compressed, shape, cr, cb)))
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    print(f"Compression time: {compress_time:.2f} ms")
    
    # Decompress to verify
    reconstructed = dct_decompress(compressed, shape, cr, cb, quality=50)
    cv2.imwrite('reconstructed.jpg', reconstructed)
    
    # Save results
    with open('image_compression_results.txt', 'w') as f:
        f.write(f"Original: {original_size} bytes\n")
        f.write(f"Compressed: {compressed_size} bytes\n")
        f.write(f"Ratio: {compression_ratio:.2f}%\n")
        f.write(f"Time: {compress_time:.2f} ms\n")