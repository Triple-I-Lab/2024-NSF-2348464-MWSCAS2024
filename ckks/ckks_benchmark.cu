#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

class CKKSBenchmark {
private:
    struct BenchmarkResult {
        string operation;
        double time_ms;
        size_t data_size;
    };
    
    vector<BenchmarkResult> results;
    
public:
    void benchmark_encoding(PhantomContext& context, const vector<double>& data, double scale) {
        PhantomCKKSEncoder encoder(context);
        PhantomPlaintext plain;
        
        auto start = chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 100; i++) {
            encoder.encode(context, data, scale, plain);
        }
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count();
        
        results.push_back({"Encode", duration / 100.0, data.size()});
    }
    
    void benchmark_encryption(PhantomContext& context, PhantomPublicKey& pk, 
                             const PhantomPlaintext& plain) {
        PhantomCiphertext cipher;
        
        auto start = chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 100; i++) {
            pk.encrypt_asymmetric(context, plain, cipher);
        }
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count();
        
        results.push_back({"Encrypt", duration / 100.0, 0});
    }
    
    void benchmark_addition(PhantomContext& context, const PhantomCiphertext& c1, 
                           const PhantomCiphertext& c2) {
        PhantomCiphertext result;
        
        auto start = chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 1000; i++) {
            result = c1;
            add_inplace(context, result, c2);
        }
        
        auto duration = chrono::duration_cast<chrono::microseconds>(
            chrono::high_resolution_clock::now() - start).count();
        
        results.push_back({"Addition", duration / 1000.0 / 1000.0, 0});
    }
    
    void benchmark_multiplication(PhantomContext& context, const PhantomCiphertext& c1, 
                                 const PhantomCiphertext& c2, PhantomRelinKey& relin) {
        PhantomCiphertext result;
        
        auto start = chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 100; i++) {
            result = phantom::multiply(context, c1, c2);
            relinearize_inplace(context, result, relin);
            rescale_to_next_inplace(context, result);
        }
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count();
        
        results.push_back({"Multiplication", duration / 100.0, 0});
    }
    
    void benchmark_decryption(PhantomContext& context, PhantomSecretKey& sk, 
                             const PhantomCiphertext& cipher) {
        PhantomPlaintext plain;
        
        auto start = chrono::high_resolution_clock::now();
        
        for(int i = 0; i < 100; i++) {
            sk.decrypt(context, cipher, plain);
        }
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(
            chrono::high_resolution_clock::now() - start).count();
        
        results.push_back({"Decrypt", duration / 100.0, 0});
    }
    
    void print_results() {
        cout << "\n=== Benchmark Results ===" << endl;
        cout << "Operation           | Time (ms)" << endl;
        cout << "--------------------|-----------" << endl;
        
        for(const auto& r : results) {
            cout << left << setw(20) << r.operation 
                 << "| " << fixed << setprecision(3) << r.time_ms << endl;
        }
    }
    
    void save_results(const string& filename) {
        ofstream file(filename);
        file << "Operation,Time_ms,Data_size\n";
        for(const auto& r : results) {
            file << r.operation << "," << r.time_ms << "," << r.data_size << "\n";
        }
        file.close();
        cout << "\nResults saved to " << filename << endl;
    }
};

int main() {
    cout << "CKKS Benchmark (Phantom-FHE GPU)" << endl;
    
    // Setup
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    
    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    
    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    double scale = pow(2.0, 40);
    
    // Test data
    vector<double> data(slot_count, 3.14);
    
    PhantomPlaintext plain;
    encoder.encode(context, data, scale, plain);
    
    PhantomCiphertext cipher1, cipher2;
    public_key.encrypt_asymmetric(context, plain, cipher1);
    public_key.encrypt_asymmetric(context, plain, cipher2);
    
    // Run benchmarks
    CKKSBenchmark benchmark;
    
    cout << "\nRunning benchmarks..." << endl;
    benchmark.benchmark_encoding(context, data, scale);
    benchmark.benchmark_encryption(context, public_key, plain);
    benchmark.benchmark_addition(context, cipher1, cipher2);
    benchmark.benchmark_multiplication(context, cipher1, cipher2, relin_keys);
    benchmark.benchmark_decryption(context, secret_key, cipher1);
    
    benchmark.print_results();
    benchmark.save_results("ckks_benchmark_results.csv");
    
    return 0;
}