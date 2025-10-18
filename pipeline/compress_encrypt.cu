#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

#define BLOCK_SIZE_2D 8
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
} while(0)

// ============================================================================
// DCT COMPRESSION KERNEL (JPEG-Style)
// ============================================================================

__global__ void dct_compress_kernel(
    const float* input,
    float* output,
    int width,
    int height,
    float threshold
) {
    int block_x = blockIdx.x * BLOCK_SIZE_2D;
    int block_y = blockIdx.y * BLOCK_SIZE_2D;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
    __shared__ float dct_block[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
    
    int x = block_x + tx;
    int y = block_y + ty;
    
    // Load 8x8 block
    if(x < width && y < height) {
        block[ty][tx] = input[y * width + x];
    } else {
        block[ty][tx] = 0.0f;
    }
    __syncthreads();
    
    // DCT-II transform
    float sum = 0.0f;
    for(int i = 0; i < BLOCK_SIZE_2D; i++) {
        for(int j = 0; j < BLOCK_SIZE_2D; j++) {
            float cos_u = cosf((2.0f * i + 1.0f) * tx * M_PI / (2.0f * BLOCK_SIZE_2D));
            float cos_v = cosf((2.0f * j + 1.0f) * ty * M_PI / (2.0f * BLOCK_SIZE_2D));
            sum += block[j][i] * cos_u * cos_v;
        }
    }
    
    // Normalization
    float alpha_u = (tx == 0) ? sqrtf(1.0f / BLOCK_SIZE_2D) : sqrtf(2.0f / BLOCK_SIZE_2D);
    float alpha_v = (ty == 0) ? sqrtf(1.0f / BLOCK_SIZE_2D) : sqrtf(2.0f / BLOCK_SIZE_2D);
    dct_block[ty][tx] = sum * alpha_u * alpha_v;
    __syncthreads();
    
    // Quantization: zero out small coefficients
    float val = dct_block[ty][tx];
    if (fabsf(val) < threshold) {
        val = 0.0f;
    }
    
    if(x < width && y < height) {
        output[y * width + x] = val;
    }
}

// ============================================================================
// DATA LOADER
// ============================================================================

vector<float> load_binary(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }
    
    file.seekg(0, ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    
    size_t num_elements = file_size / sizeof(float);
    vector<float> data(num_elements);
    
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    
    cout << "Loaded " << num_elements << " floats from " << filename << endl;
    return data;
}

// ============================================================================
// DCT COMPRESSOR CLASS
// ============================================================================

class DCTCompressor {
private:
    float* d_input;
    float* d_output;
    int width, height;
    
public:
    DCTCompressor(int w, int h) : width(w), height(h) {
        CUDA_CHECK(cudaMalloc(&d_input, w * h * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, w * h * sizeof(float)));
    }
    
    ~DCTCompressor() {
        cudaFree(d_input);
        cudaFree(d_output);
    }
    
    vector<float> compress(const vector<float>& data, float quality, double& time_ms) {
        CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Calculate threshold: higher quality = lower threshold = less compression
        float threshold = 0.5f * (1.0f - quality) * 50.0f;
        
        dim3 grid((width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D, 
                  (height + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
        dim3 block(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
        
        auto start = chrono::high_resolution_clock::now();
        
        dct_compress_kernel<<<grid, block>>>(d_input, d_output, width, height, threshold);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = chrono::high_resolution_clock::now();
        time_ms = chrono::duration<double, milli>(end - start).count();
        
        vector<float> result(data.size());
        CUDA_CHECK(cudaMemcpy(result.data(), d_output, data.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        return result;
    }
};

// ============================================================================
// MAIN PIPELINE
// ============================================================================

int main(int argc, char** argv) {
    cout << "\n========================================" << endl;
    cout << "DCT Compress + Encrypt Pipeline" << endl;
    cout << "========================================\n" << endl;
    
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <data_file> [quality]" << endl;
        cout << "  quality: 0.1-0.9 (default 0.5)" << endl;
        cout << "  0.1 = high compression, 0.9 = low compression" << endl;
        cout << "\nExample: " << argv[0] << " data/image_512x512.bin 0.5" << endl;
        return 1;
    }
    
    string data_file = argv[1];
    float quality = (argc > 2) ? atof(argv[2]) : 0.5f;
    
    // Load data
    vector<float> data = load_binary(data_file);
    size_t original_size = data.size() * sizeof(float);
    
    // Make square for DCT
    int dim = (int)sqrt(data.size());
    if (dim * dim != data.size()) {
        dim = (int)ceil(sqrt(data.size()));
        data.resize(dim * dim, 0.0f);
    }
    
    cout << "Data: " << dim << "x" << dim << " (" << original_size << " bytes)" << endl;
    cout << "Quality: " << quality << endl;
    
    // Step 1: DCT Compression
    cout << "\n--- Compression ---" << endl;
    DCTCompressor compressor(dim, dim);
    double comp_time;
    vector<float> compressed = compressor.compress(data, quality, comp_time);
    
    // Count non-zero coefficients (sparse representation)
    int non_zero = 0;
    for (float val : compressed) {
        if (val != 0.0f) non_zero++;
    }
    
    size_t compressed_size = non_zero * 2 * sizeof(int);  // Store (position, value) pairs
    double comp_ratio = (1.0 - (double)compressed_size / original_size) * 100;
    
    cout << "Compression time: " << comp_time << " ms" << endl;
    cout << "Non-zero coefficients: " << non_zero << " / " << compressed.size() << endl;
    cout << "Compressed size: " << compressed_size << " bytes" << endl;
    cout << "Compression ratio: " << comp_ratio << "%" << endl;
    
    // Convert to sparse representation
    vector<double> sparse_data;
    sparse_data.reserve(non_zero * 2);
    for (size_t i = 0; i < compressed.size(); i++) {
        if (compressed[i] != 0.0f) {
            sparse_data.push_back((double)i);  // Position
            sparse_data.push_back((double)(compressed[i] * 100.0f));  // Quantized value
        }
    }
    
    cout << "Sparse data size: " << sparse_data.size() << " values" << endl;
    
    // Step 2: Encryption
    cout << "\n--- Encryption ---" << endl;
    
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(CoeffModulus::Create(8192, {60, 40, 40, 60}));
    PhantomContext context(parms);
    
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);
    double scale = pow(2.0, 40);
    
    cout << "Slot count: " << encoder.slot_count() << endl;
    
    // Pad to slot count
    if (sparse_data.size() > encoder.slot_count()) {
        cout << "Warning: Data larger than slot count, truncating" << endl;
        sparse_data.resize(encoder.slot_count());
    } else if (sparse_data.size() < encoder.slot_count()) {
        sparse_data.resize(encoder.slot_count(), 0.0);
    }
    
    PhantomPlaintext plain;
    encoder.encode(context, sparse_data, scale, plain);
    
    auto start = chrono::high_resolution_clock::now();
    PhantomCiphertext cipher;
    public_key.encrypt_asymmetric(context, plain, cipher);
    auto enc_time = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - start).count();
    cout << "Encryption time: " << enc_time << " ms" << endl;
    
    // Step 3: Decryption
    cout << "\n--- Decryption ---" << endl;
    start = chrono::high_resolution_clock::now();
    PhantomPlaintext decrypted;
    secret_key.decrypt(context, cipher, decrypted);
    auto dec_time = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - start).count();
    cout << "Decryption time: " << dec_time << " ms" << endl;
    
    vector<double> decoded;
    encoder.decode(context, decrypted, decoded);
    
    // Verify
    double max_error = 0;
    for (size_t i = 0; i < min(sparse_data.size(), decoded.size()); i++) {
        max_error = max(max_error, abs(sparse_data[i] - decoded[i]));
    }
    
    cout << "\n--- Summary ---" << endl;
    cout << "Max reconstruction error: " << max_error << endl;
    cout << "Total time: " << (comp_time + enc_time + dec_time) << " ms" << endl;
    
    // Save results
    ofstream out("dct_results.csv");
    out << "Metric,Value\n";
    out << "Original_size_bytes," << original_size << "\n";
    out << "Compressed_size_bytes," << compressed_size << "\n";
    out << "Compression_ratio_percent," << comp_ratio << "\n";
    out << "Non_zero_coefficients," << non_zero << "\n";
    out << "Compression_time_ms," << comp_time << "\n";
    out << "Encryption_time_ms," << enc_time << "\n";
    out << "Decryption_time_ms," << dec_time << "\n";
    out << "Total_time_ms," << (comp_time + enc_time + dec_time) << "\n";
    out << "Max_error," << max_error << "\n";
    out.close();
    
    cout << "\nResults saved to dct_results.csv" << endl;
    
    return 0;
}