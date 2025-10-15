#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define DICT_SIZE 4096
#define BLOCK_SIZE 256

// GPU kernel for dictionary-based compression
__global__ void dcb_compress_kernel(
    const char* input, 
    int* output, 
    int* dict, 
    int input_size,
    int* output_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ int local_dict[DICT_SIZE];
    
    // Initialize local dictionary
    if(threadIdx.x < 256) {
        local_dict[threadIdx.x] = threadIdx.x;
    }
    __syncthreads();
    
    // Each thread compresses its chunk
    for(int i = tid; i < input_size; i += stride) {
        // Simple compression: map character to code
        if(i < input_size) {
            int code = (int)input[i];
            if(code >= 0 && code < 256) {
                output[i] = local_dict[code];
            }
        }
    }
}

class DCBCompressor {
private:
    char* d_input;
    int* d_output;
    int* d_dict;
    size_t input_size;
    
public:
    DCBCompressor(size_t size) : input_size(size) {
        cudaMalloc(&d_input, size * sizeof(char));
        cudaMalloc(&d_output, size * sizeof(int));
        cudaMalloc(&d_dict, DICT_SIZE * sizeof(int));
    }
    
    ~DCBCompressor() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_dict);
    }
    
    std::vector<int> compress(const std::string& data) {
        // Copy input to GPU
        cudaMemcpy(d_input, data.c_str(), data.size(), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int num_blocks = (data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        dcb_compress_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_input, d_output, d_dict, data.size(), nullptr
        );
        
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "GPU Compression time: " << duration.count() << " ms" << std::endl;
        
        // Copy result back
        std::vector<int> result(data.size());
        cudaMemcpy(result.data(), d_output, data.size() * sizeof(int), cudaMemcpyDeviceToHost);
        
        return result;
    }
};

int main(int argc, char** argv) {
    std::cout << "DCB Compression on GPU" << std::endl;
    
    // Test data
    std::string test_data(10000, 'A');
    for(int i = 0; i < 10000; i++) {
        test_data[i] = 'A' + (i % 26);
    }
    
    std::cout << "Input size: " << test_data.size() << " bytes" << std::endl;
    
    DCBCompressor compressor(test_data.size());
    auto compressed = compressor.compress(test_data);
    
    std::cout << "Compressed size: " << compressed.size() * sizeof(int) << " bytes" << std::endl;
    std::cout << "Compression ratio: " 
              << (1.0 - (compressed.size() * sizeof(int)) / (double)test_data.size()) * 100 
              << "%" << std::endl;
    
    return 0;
}