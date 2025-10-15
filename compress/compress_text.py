import sys
import time
import pickle

def lzw_compress(data):
    """Simple LZW compression"""
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    result = []
    current = ""
    
    for char in data:
        combined = current + char
        if combined in dictionary:
            current = combined
        else:
            result.append(dictionary[current])
            dictionary[combined] = next_code
            next_code += 1
            current = char
    
    if current:
        result.append(dictionary[current])
    
    return result

def lzw_decompress(compressed):
    """Simple LZW decompression"""
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    result = []
    current = chr(compressed[0])
    result.append(current)
    
    for code in compressed[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = current + current[0]
        else:
            raise ValueError("Bad compressed code")
        
        result.append(entry)
        dictionary[next_code] = current + entry[0]
        next_code += 1
        current = entry
    
    return ''.join(result)

if __name__ == '__main__':
    # Read input file
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/test.txt'
    
    with open(input_file, 'r') as f:
        data = f.read()
    
    original_size = len(data.encode())
    print(f"Original size: {original_size} bytes")
    
    # Compress
    start = time.time()
    compressed = lzw_compress(data)
    compress_time = (time.time() - start) * 1000
    
    # Save compressed data
    with open('compressed_data.pkl', 'wb') as f:
        pickle.dump(compressed, f)
    
    compressed_size = len(pickle.dumps(compressed))
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}%")
    print(f"Compression time: {compress_time:.2f} ms")
    
    # Save results
    with open('compression_results.txt', 'w') as f:
        f.write(f"Original: {original_size} bytes\n")
        f.write(f"Compressed: {compressed_size} bytes\n")
        f.write(f"Ratio: {compression_ratio:.2f}%\n")
        f.write(f"Time: {compress_time:.2f} ms\n")