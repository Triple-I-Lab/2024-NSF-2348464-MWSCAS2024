# GPU-Accelerated CKKS with Data Compression

Implementation of "Acceleration of CKKS Algorithm with GPU-Driven and Data Compression" achieving **100x speedup** and **90% compression**.

**Paper**: IEEE MWSCAS 2024 - Phan et al.

Built on [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe) GPU library.

## Quick Start
```bash
# Generate test data
python3 python_scripts/generate_test_data.py --type all

# Build CUDA code
cd build
cmake ..
make -j

# Run experiments
./ckks_operations
./ckks_benchmark
./compress_encrypt 8192

# Visualize results
python3 python_scripts/visualize_results.py --type all
```

## Project Structure
```
├── compress/              # GPU compression (DCB, DCT)
├── ckks/                 # CKKS operations & benchmarks
├── pipeline/             # Full compress → encrypt pipeline
├── python_scripts/       # Data generation & visualization
└── data/                 # Test datasets
```

## Key Results

| Operation     | Time (ms) | Speedup vs SEAL |
|---------------|-----------|-----------------|
| Encryption    | 0.453     | 66x             |
| Rotation      | 3.112     | 12x             |
| Decode        | 1.319     | 13x             |

**Compression**: 90% for images, 49% for text

## Requirements

- NVIDIA GPU with CUDA 12.3+
- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe)
- Python 3.8+ (matplotlib, pandas, numpy, pillow)

## Citation
```bibtex
@inproceedings{phan2024acceleration,
  title={Acceleration of CKKS Algorithm with GPU-Driven and Data Compression},
  author={Phan, Quoc Bao and Nguyen, Linh and Nguyen, Tuy Tan},
  booktitle={IEEE MWSCAS},
  year={2024}
}
```

## Acknowledgments

- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe) for GPU-accelerated HE primitives
- NSF Grant No. 2348464

## License

MIT