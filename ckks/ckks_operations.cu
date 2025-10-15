#include <iostream>
#include <vector>
#include <chrono>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

class CKKSOperations {
private:
    PhantomContext context;
    PhantomSecretKey secret_key;
    PhantomPublicKey public_key;
    PhantomRelinKey relin_keys;
    PhantomGaloisKey galois_keys;
    PhantomCKKSEncoder encoder;
    double scale;
    
public:
    CKKSOperations(size_t poly_modulus_degree = 8192, double s = pow(2.0, 40)) 
        : context(create_context(poly_modulus_degree)), 
          secret_key(context),
          public_key(secret_key.gen_publickey(context)),
          relin_keys(secret_key.gen_relinkey(context)),
          galois_keys(secret_key.create_galois_keys(context)),
          encoder(context),
          scale(s) {
        
        cout << "CKKS initialized with poly_modulus_degree: " << poly_modulus_degree << endl;
        cout << "Slot count: " << encoder.slot_count() << endl;
    }
    
    static PhantomContext create_context(size_t poly_modulus_degree) {
        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
        return PhantomContext(parms);
    }
    
    // Encode data
    PhantomPlaintext encode(const vector<double>& data) {
        PhantomPlaintext plain;
        encoder.encode(context, data, scale, plain);
        return plain;
    }
    
    // Decode plaintext
    vector<double> decode(const PhantomPlaintext& plain) {
        vector<double> result;
        encoder.decode(context, plain, result);
        return result;
    }
    
    // Encrypt plaintext
    PhantomCiphertext encrypt(const PhantomPlaintext& plain) {
        PhantomCiphertext cipher;
        public_key.encrypt_asymmetric(context, plain, cipher);
        return cipher;
    }
    
    // Decrypt ciphertext
    PhantomPlaintext decrypt(const PhantomCiphertext& cipher) {
        PhantomPlaintext plain;
        secret_key.decrypt(context, cipher, plain);
        return plain;
    }
    
    // Homomorphic addition
    PhantomCiphertext add(const PhantomCiphertext& c1, const PhantomCiphertext& c2) {
        PhantomCiphertext result = c1;
        add_inplace(context, result, c2);
        return result;
    }
    
    // Homomorphic subtraction
    PhantomCiphertext subtract(const PhantomCiphertext& c1, const PhantomCiphertext& c2) {
        PhantomCiphertext result = c1;
        sub_inplace(context, result, c2);
        return result;
    }
    
    // Homomorphic multiplication
    PhantomCiphertext multiply(const PhantomCiphertext& c1, const PhantomCiphertext& c2) {
        PhantomCiphertext result = phantom::multiply(context, c1, c2);
        relinearize_inplace(context, result, relin_keys);
        rescale_to_next_inplace(context, result);
        return result;
    }
    
    // Homomorphic rotation
    PhantomCiphertext rotate(const PhantomCiphertext& cipher, int steps) {
        PhantomCiphertext result = cipher;
        rotate_inplace(context, result, steps, galois_keys);
        return result;
    }
    
    size_t get_slot_count() const {
        return encoder.slot_count();
    }
};

// Test function
void test_ckks_operations() {
    cout << "\n=== Testing CKKS Operations ===" << endl;
    
    CKKSOperations ckks;
    size_t slot_count = ckks.get_slot_count();
    
    // Test data
    vector<double> input1(slot_count, 3.14);
    vector<double> input2(slot_count, 2.71);
    
    cout << "\nInput 1: [" << input1[0] << ", " << input1[1] << ", ...]" << endl;
    cout << "Input 2: [" << input2[0] << ", " << input2[1] << ", ...]" << endl;
    
    // Encode
    auto start = chrono::high_resolution_clock::now();
    auto plain1 = ckks.encode(input1);
    auto plain2 = ckks.encode(input2);
    auto encode_time = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Encrypt
    start = chrono::high_resolution_clock::now();
    auto cipher1 = ckks.encrypt(plain1);
    auto cipher2 = ckks.encrypt(plain2);
    auto encrypt_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Addition
    start = chrono::high_resolution_clock::now();
    auto cipher_add = ckks.add(cipher1, cipher2);
    auto add_time = chrono::duration_cast<chrono::microseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Multiplication
    start = chrono::high_resolution_clock::now();
    auto cipher_mul = ckks.multiply(cipher1, cipher2);
    auto mul_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Decrypt
    start = chrono::high_resolution_clock::now();
    auto decrypted_add = ckks.decrypt(cipher_add);
    auto decrypted_mul = ckks.decrypt(cipher_mul);
    auto decrypt_time = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Decode
    auto result_add = ckks.decode(decrypted_add);
    auto result_mul = ckks.decode(decrypted_mul);
    
    cout << "\nResults:" << endl;
    cout << "Addition: " << result_add[0] << " (expected: " << input1[0] + input2[0] << ")" << endl;
    cout << "Multiplication: " << result_mul[0] << " (expected: " << input1[0] * input2[0] << ")" << endl;
    
    cout << "\nPerformance:" << endl;
    cout << "Encode time: " << encode_time << " us" << endl;
    cout << "Encrypt time: " << encrypt_time << " ms" << endl;
    cout << "Addition time: " << add_time << " us" << endl;
    cout << "Multiplication time: " << mul_time << " ms" << endl;
    cout << "Decrypt time: " << decrypt_time << " ms" << endl;
}

int main() {
    cout << "CKKS Operations on GPU (Phantom-FHE)" << endl;
    
    try {
        test_ckks_operations();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}