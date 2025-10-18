#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv) < 3:
    print("Usage: python3 verify.py <original.bin> <decompressed.bin>")
    sys.exit(1)

original_file = sys.argv[1]
decompressed_file = sys.argv[2]

# Load both files
original = np.fromfile(original_file, dtype=np.float32)
decompressed = np.fromfile(decompressed_file, dtype=np.float32)

print(f"Original size: {len(original)}")
print(f"Decompressed size: {len(decompressed)}")

# Check size match
if len(original) != len(decompressed):
    print("❌ FAILED: Size mismatch!")
    sys.exit(1)

# Calculate differences
diff = np.abs(original - decompressed)
max_error = np.max(diff)
mean_error = np.mean(diff)
mse = np.mean((original - decompressed) ** 2)
psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

print(f"\n=== Verification Results ===")
print(f"Max error: {max_error:.6f}")
print(f"Mean error: {mean_error:.6f}")
print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.2f} dB")

# Check if identical or within tolerance
if max_error == 0:
    print("✅ PERFECT: Files are identical!")
elif max_error < 1.0:
    print(f"✅ GOOD: Max error < 1.0 (acceptable for image compression)")
else:
    print(f"⚠️  WARNING: Max error = {max_error:.2f}")
    
# Show first 10 values comparison
print(f"\nFirst 10 values comparison:")
print(f"Original:      {original[:10]}")
print(f"Decompressed:  {decompressed[:10]}")