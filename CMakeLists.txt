cmake_minimum_required(VERSION 3.18)
project(ckks_gpu_compression CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Phantom-FHE
set(PHANTOM_DIR $ENV{HOME}/workspace/phantom-fhe)
include_directories(${PHANTOM_DIR}/include)
include_directories(${PHANTOM_DIR}/src)
link_directories(${PHANTOM_DIR}/build)

# CKKS operations
add_executable(ckks_operations ckks/ckks_operations.cu)
target_link_libraries(ckks_operations Phantom cudart)
set_target_properties(ckks_operations PROPERTIES BUILD_RPATH "${PHANTOM_DIR}/build")

# CKKS benchmark
add_executable(ckks_benchmark ckks/ckks_benchmark.cu)
target_link_libraries(ckks_benchmark Phantom cudart)
set_target_properties(ckks_benchmark PROPERTIES BUILD_RPATH "${PHANTOM_DIR}/build")

# Full pipeline
add_executable(compress_encrypt pipeline/compress_encrypt.cu)
target_link_libraries(compress_encrypt Phantom cudart)
set_target_properties(compress_encrypt PROPERTIES BUILD_RPATH "${PHANTOM_DIR}/build")