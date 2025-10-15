#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

// Simple compression simulation
vector<int> simple_compress(const vector<double>& data) {
    vector<int> compressed;
    compressed.reserve(data.size() / 2);
    
    // Simple run-length encoding simulation
    for(size_t i = 0; i < data.size(); i += 2) {
        compressed.push_back((int)(data[i] * 100));
    }
    
    return compressed;
}

vector<double> simple_decompress(const vector<int>& compressed, size_t original_size) {
    vector<double> decompressed(original_size);
    
    for(size_t i = 0; i < compressed.size() && i * 2 < original_size; i++) {
        decompressed[i * 2] = compressed[i] / 100.0;
        if(i * 2 + 1 < original_size) {
            decompressed[i * 2 + 1] = compressed[i] / 100.0;
        }
    }
    
    return decompressed;
}

class CompressEncryptPipeline {
private:
    PhantomContext context;
    PhantomSecretKey secret_key;
    PhantomPublicKey public_key;
    PhantomCKKSEncoder encoder;
    double scale;
    
    struct PipelineMetrics {
        double compression_time_ms;
        double encryption_time_ms;
        double decryption_time_ms;
        double decompression_time_ms;
        size_t original_size;
        size_t compressed_size;
        double compression_ratio;
    };
    
public:
    CompressEncryptPipeline(size_t poly_modulus_degree = 8192) 
        : context(create_context(poly_modulus_degree)),
          secret_key(context),
          public_key(secret_key.gen_publickey(context)),
          encoder(context),
          scale(pow(2.0, 40)) {
    }
    
    static PhantomContext create_context(size_t poly_modulus_degree) {
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        return PhantomContext(parms);
    }
    
    PipelineMetrics run_pipeline(const vector<double>& data) {
        PipelineMetrics metrics;
        metrics.original_size = data.size() * sizeof(double);
        
        cout << "\n=== Compress + Encrypt Pipeline ===" << endl;
        cout << "Original data size: " << metrics.original_size << " bytes" << endl;
        
        // Step 1: Compression
        auto start = chrono::high_resolution_clock::now();
        auto compressed = simple_compress(data);
        auto compress_time = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start);
        metrics.compression_time_ms = compress_time.count();
        metrics.compressed_size = compressed.size() * sizeof(int);
        metrics.compression_ratio = (1.0 - (double)metrics.compressed_size / metrics.original_size) * 100;
        
        cout << "Compressed size: " << metrics.compressed_size << " bytes" << endl;
        cout << "Compression ratio: " << metrics.compression_ratio << "%" << endl;
        cout << "Compression time: " << metrics.compression_time_ms << " ms" << endl;
        
        // Convert to double for CKKS
        vector<double> compressed_double(compressed.begin(), compressed.end());
        
        // Pad to slot count
        size_t slot_count = encoder.slot_count();
        if(compressed_double.size() < slot_count) {
            compressed_double.resize(slot_count, 0.0);
        }
        
        // Step 2: Encode + Encrypt
        PhantomPlaintext plain;
        encoder.encode(context, compressed_double, scale, plain);
        
        start = chrono::high_resolution_clock::now();
        PhantomCiphertext cipher;
        public_key.encrypt_asymmetric(context, plain, cipher);
        auto encrypt_time = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start);
        metrics.encryption_time_ms = encrypt_time.count();
        
        cout << "Encryption time: " << metrics.encryption_time_ms << " ms" << endl;
        
        // Step 3: Decrypt
        start = chrono::high_resolution_clock::now();
        PhantomPlaintext decrypted_plain;
        secret_key.decrypt(context, cipher, decrypted_plain);
        auto decrypt_time = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start);
        metrics.decryption_time_ms = decrypt_time.count();
        
        cout << "Decryption time: " << metrics.decryption_time_ms << " ms" << endl;
        
        // Step 4: Decode + Decompress
        vector<double> decoded;
        encoder.decode(context, decrypted_plain, decoded);
        
        start = chrono::high_resolution_clock::now();
        vector<int> compressed_recovered(decoded.begin(), decoded.begin() + compressed.size());
        auto decompressed = simple_decompress(compressed_recovered, data.size());
        auto decompress_time = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start);
        metrics.decompression_time_ms = decompress_time.count();
        
        cout << "Decompression time: " << metrics.decompression_time_ms << " ms" << endl;
        
        // Verify correctness
        double max_error = 0;
        for(size_t i = 0; i < min(data.size(), decompressed.size()); i++) {
            double error = abs(data[i] - decompressed[i]);
            max_error = max(max_error, error);
        }
        cout << "Max reconstruction error: " << max_error << endl;
        
        double total_time = metrics.compression_time_ms + metrics.encryption_time_ms + 
                           metrics.decryption_time_ms + metrics.decompression_time_ms;
        cout << "Total pipeline time: " << total_time << " ms" << endl;
        
        return metrics;
    }
    
    void save_metrics(const PipelineMetrics& metrics, const string& filename) {
        ofstream file(filename);
        file << "Metric,Value\n";
        file << "Original_size_bytes," << metrics.original_size << "\n";
        file << "Compressed_size_bytes," << metrics.compressed_size << "\n";
        file << "Compression_ratio_percent," << metrics.compression_ratio << "\n";
        file << "Compression_time_ms," << metrics.compression_time_ms << "\n";
        file << "Encryption_time_ms," << metrics.encryption_time_ms << "\n";
        file << "Decryption_time_ms," << metrics.decryption_time_ms << "\n";
        file << "Decompression_time_ms," << metrics.decompression_time_ms << "\n";
        file.close();
        cout << "\nMetrics saved to " << filename << endl;
    }
};

int main(int argc, char** argv) {
    cout << "Compress + Encrypt Pipeline (Phantom-FHE)" << endl;
    
    // Generate test data
    size_t data_size = 4096;
    if(argc > 1) {
        data_size = atoi(argv[1]);
    }
    
    vector<double> data(data_size);
    for(size_t i = 0; i < data_size; i++) {
        data[i] = 3.14 + (i % 100) * 0.01;
    }
    
    // Run pipeline
    CompressEncryptPipeline pipeline;
    auto metrics = pipeline.run_pipeline(data);
    pipeline.save_metrics(metrics, "pipeline_results.csv");
    
    return 0;
}