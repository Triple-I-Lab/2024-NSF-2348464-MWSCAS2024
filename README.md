# GPU-Accelerated CKKS with Data Compression

Implementation of "Acceleration of CKKS Algorithm with GPU-Driven and Data Compression" achieving **100x speedup** and **90% compression**.

**Paper**: IEEE MWSCAS 2024 - Phan et al.

Built on [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe) GPU library.

## Quick Start

### 1. Generate Test Data
```bash
python3 python_scripts/generate_test_data.py
```

### 2. Build
```bash
mkdir -p build && cd build
cmake ..
make -j
```

### 3. Run Experiments

#### CKKS Operations
```bash
./ckks_operations          # Basic CKKS operations demo
./ckks_benchmark           # Performance benchmarks
```

#### Full Pipeline
```bash
./compress_encrypt ../data/sensor_data.bin
./compress_encrypt ../data/image_512x512.bin
```

#### Pipeline
```bash
python3 ../compress/compress.py ../data/sensor_data.bin text
./encrypt_decrypt ../data/sensor_data_compressed.bin
python3 ../compress/decompress.py ../data/sensor_data_encrypted.bin

python3 ../compress/compress.py ../data/image_64x64.bin image
./encrypt_decrypt ../data/image_64x64_compressed.bin
python3 ../compress/decompress.py ../data/image_64x64_encrypted.bin
```

#### Verify Results
```bash
python3 ../python_scripts/verify.py ../data/{original.bin} ../data/{decompressed_file.bin}
```

## Project Structure
```
├── compress/              # GPU compression + Python hybrid
│   ├── dcb_compression.cu
│   ├── dct_compression.cu
│   ├── compress.py        # Python compression (text/image)
│   └── decompress.py      # Python decompression
├── ckks/                  # CKKS operations & benchmarks
│   ├── ckks_operations.cu
│   └── ckks_benchmark.cu
├── pipeline/              # Full pipelines
│   ├── compress_encrypt.cu  # All-in-one CUDA pipeline
│   └── enc_dec.cu           # Encrypt/decrypt only
├── python_scripts/        # Data generation & visualization
│   ├── generate_test_data.py
│   └── verify.py
└── data/                  # Test datasets
```

## Requirements

- CUDA Toolkit 11.0+
- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe)
- Python 3.11+ with: `pip install numpy pillow`

## Citation
```bibtex
@inproceedings{phan2024acceleration,
  title={Acceleration of CKKS Algorithm with GPU-Driven and Data Compression},
  author={Phan, Quoc Bao and Nguyen, Linh and Nguyen, Tuy Tan},
  booktitle={67th International Midwest Symposium on Circuits and Systems (MWSCAS)},
  pages={1145--1149},
  year={2024},
  organization={IEEE}
}
```

## Acknowledgments

- [Phantom-FHE](https://github.com/encryptorion-lab/phantom-fhe) for GPU-accelerated HE primitives
- NSF Grant No. 2348464

## License


MIT

