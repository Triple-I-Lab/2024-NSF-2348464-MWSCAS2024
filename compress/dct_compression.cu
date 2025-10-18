#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define BLOCK_SIZE_2D 8

// GPU kernel for DCT on 8x8 blocks
__global__ void dct_compress_kernel(
    float* input,
    float* output,
    int width,
    int height
) {
    int block_x = blockIdx.x * BLOCK_SIZE_2D;
    int block_y = blockIdx.y * BLOCK_SIZE_2D;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
    __shared__ float dct_block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
    
    // Load 8x8 block
    int x = block_x + tx;
    int y = block_y + ty;
    
    if(x < width && y < height) {
        block[ty][tx] = input[y * width + x];
    } else {
        block[ty][tx] = 0;
    }
    __syncthreads();
    
    // DCT transform
    float sum = 0;
    for(int i = 0; i < BLOCK_SIZE_2D; i++) {
        for(int j = 0; j < BLOCK_SIZE_2D; j++) {
            float cos_u = cosf((2*i + 1) * tx * M_PI / (2.0 * BLOCK_SIZE_2D));
            float cos_v = cosf((2*j + 1) * ty * M_PI / (2.0 * BLOCK_SIZE_2D));
            sum += block[j][i] * cos_u * cos_v;
        }
    }
    
    dct_block[ty][tx] = sum / 4.0;
    __syncthreads();
    
    // Write back
    if(x < width && y < height) {
        output[y * width + x] = dct_block[ty][tx];
    }
}

class DCTCompressor {
private:
    float* d_input;
    float* d_output;
    int width, height;
    
public:
    DCTCompressor(int w, int h) : width(w), height(h) {
        cudaMalloc(&d_input, w * h * sizeof(float));
        cudaMalloc(&d_output, w * h * sizeof(float));
    }
    
    ~DCTCompressor() {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    std::vector<float> compress(const std::vector<float>& image) {
        cudaMemcpy(d_input, image.data(), image.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        dim3 grid((width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 
                  (height + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
        dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        dct_compress_kernel<<<grid, block>>>(d_input, d_output, width, height);
        
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "DCT GPU time: " << duration.count() << " ms" << std::endl;
        
        std::vector<float> result(image.size());
        cudaMemcpy(result.data(), d_output, image.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        return result;
    }
};

int main() {
    std::cout << "DCT Compression on GPU" << std::endl;
    
    int width = 512, height = 512;
    std::vector<float> image(width * height);
    
    // Generate test image
    for(int i = 0; i < width * height; i++) {
        image[i] = rand() % 256;
    }
    
    DCTCompressor compressor(width, height);
    auto compressed = compressor.compress(image);
    
    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Compression complete" << std::endl;
    
    return 0;
}