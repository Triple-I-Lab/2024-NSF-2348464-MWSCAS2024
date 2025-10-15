# GPU-Accelerated CKKS with Data Compression

Implementation of "Acceleration of CKKS Algorithm with GPU-Driven and Data Compression" achieving **100x speedup** and **90% compression**.

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

## Requirements

- NVIDIA GPU with CUDA 12.3+
- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe)
- Python 3.8+ (matplotlib, pandas, numpy, pillow)

## Citation
```bibtex
@inproceedings{phan2024accelerating,
  title={Accelerating CKKS Homomorphic Encryption with Data Compression on GPUs},
  author={Phan, Quoc Bao and Nguyen, Linh and Nguyen, Tuy Tan},
  booktitle={67th International Midwest Symposium on Circuits and Systems (MWSCAS)},
  pages={1145--1149},
  address={Springfield, MA},
  month={Aug},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgments

- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe) for GPU-accelerated HE primitives
- NSF Grant No. 2348464

## License


MIT
