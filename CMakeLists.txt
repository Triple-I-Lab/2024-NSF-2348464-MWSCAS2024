cmake_minimum_required(VERSION 3.18)
project(ckks_gpu_compression CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# CUDA architecture for GTX 1650
set(CMAKE_CUDA_ARCHITECTURES 75)

# Phantom-FHE configuration
set(PHANTOM_DIR $ENV{HOME}/phantom-fhe)
if(NOT EXISTS ${PHANTOM_DIR})
    set(PHANTOM_DIR $ENV{HOME}/workspace/phantom-fhe)
endif()

if(NOT EXISTS ${PHANTOM_DIR})
    message(FATAL_ERROR "PhantomFHE not found. Please set PHANTOM_DIR or install at ~/phantom-fhe")
endif()

message(STATUS "Using PhantomFHE from: ${PHANTOM_DIR}")

# Include directories
include_directories(
    ${PHANTOM_DIR}/include
    ${PHANTOM_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link directories
link_directories(
    ${PHANTOM_DIR}/build
    ${PHANTOM_DIR}/build/lib
)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# ============================================================================
# CKKS Operations
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ckks/ckks_operations.cu)
    add_executable(ckks_operations ckks/ckks_operations.cu)
    target_link_libraries(ckks_operations 
        Phantom 
        cudart
    )
    set_target_properties(ckks_operations PROPERTIES 
        BUILD_RPATH "${PHANTOM_DIR}/build"
        INSTALL_RPATH "${PHANTOM_DIR}/build"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# CKKS Benchmark
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ckks/ckks_benchmark.cu)
    add_executable(ckks_benchmark ckks/ckks_benchmark.cu)
    target_link_libraries(ckks_benchmark 
        Phantom 
        cudart
    )
    set_target_properties(ckks_benchmark PROPERTIES 
        BUILD_RPATH "${PHANTOM_DIR}/build"
        INSTALL_RPATH "${PHANTOM_DIR}/build"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# Compression Modules
# ============================================================================
# DCB Compression
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compress/dcb_compression.cu)
    add_executable(dcb_compression compress/dcb_compression.cu)
    target_link_libraries(dcb_compression cudart)
    set_target_properties(dcb_compression PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# DCT Compression
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compress/dct_compression.cu)
    add_executable(dct_compression compress/dct_compression.cu)
    target_link_libraries(dct_compression cudart)
    set_target_properties(dct_compression PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# Full Compress + Encrypt Pipeline
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/compress_encrypt.cu)
    add_executable(compress_encrypt pipeline/compress_encrypt.cu)
    target_link_libraries(compress_encrypt 
        Phantom 
        cudart
    )
    set_target_properties(compress_encrypt PROPERTIES 
        BUILD_RPATH "${PHANTOM_DIR}/build"
        INSTALL_RPATH "${PHANTOM_DIR}/build"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Alternative location for compress_encrypt.cu (root directory)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compress_encrypt.cu)
    add_executable(compress_encrypt_root compress_encrypt.cu)
    target_link_libraries(compress_encrypt_root 
        Phantom 
        cudart
    )
    set_target_properties(compress_encrypt_root PROPERTIES 
        BUILD_RPATH "${PHANTOM_DIR}/build"
        INSTALL_RPATH "${PHANTOM_DIR}/build"
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME compress_encrypt
    )
endif()

# ============================================================================
# Encrypt/Decrypt Only (for pre-compressed data)
# ============================================================================
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/enc_dec.cu)
    add_executable(encrypt_decrypt pipeline/enc_dec.cu)
    target_link_libraries(encrypt_decrypt 
        Phantom 
        cudart
    )
    set_target_properties(encrypt_decrypt PROPERTIES 
        BUILD_RPATH "${PHANTOM_DIR}/build"
        INSTALL_RPATH "${PHANTOM_DIR}/build"
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)