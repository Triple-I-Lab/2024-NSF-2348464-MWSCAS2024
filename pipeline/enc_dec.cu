#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./encrypt_decrypt <compressed.bin>" << endl;
        return 1;
    }
    
    // Load compressed data
    ifstream file(argv[1], ios::binary);
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    
    vector<uint8_t> compressed(size);
    file.read((char*)compressed.data(), size);
    file.close();
    
    cout << "Loaded: " << size << " bytes" << endl;
    
    // Pack 4 uint8 into 1 double (32-bit integer range)
    // This avoids floating point precision issues
    vector<double> packed;
    for (size_t i = 0; i < compressed.size(); i += 4) {
        uint32_t value = 0;
        for (int j = 0; j < 4 && i + j < compressed.size(); j++) {
            value |= ((uint32_t)compressed[i + j]) << (j * 8);
        }
        packed.push_back((double)value);
    }
    
    cout << "Packed: " << size << " bytes -> " << packed.size() << " values" << endl;
    
    // Setup CKKS
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(8192);
    parms.set_coeff_modulus(CoeffModulus::Create(8192, {60, 40, 40, 60}));
    PhantomContext context(parms);
    
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomCKKSEncoder encoder(context);
    double scale = pow(2.0, 40);
    
    size_t slot_count = encoder.slot_count();
    cout << "Slot count: " << slot_count << endl;
    
    // Pad to slot count
    if (packed.size() > slot_count) {
        cout << "Error: Data too large" << endl;
        return 1;
    }
    packed.resize(slot_count, 0.0);
    
    // Encode
    PhantomPlaintext plain;
    encoder.encode(context, packed, scale, plain);
    
    // Encrypt
    auto start = chrono::high_resolution_clock::now();
    PhantomCiphertext cipher;
    public_key.encrypt_asymmetric(context, plain, cipher);
    auto enc_time = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - start).count();
    cout << "Encryption: " << enc_time << " ms" << endl;
    
    // Decrypt
    start = chrono::high_resolution_clock::now();
    PhantomPlaintext decrypted;
    secret_key.decrypt(context, cipher, decrypted);
    auto dec_time = chrono::duration<double, milli>(chrono::high_resolution_clock::now() - start).count();
    cout << "Decryption: " << dec_time << " ms" << endl;
    
    // Decode
    vector<double> decoded;
    encoder.decode(context, decrypted, decoded);
    decoded.resize(packed.size());
    
    // Unpack back to uint8
    vector<uint8_t> output;
    output.reserve(size);
    for (size_t i = 0; i < decoded.size() && output.size() < size; i++) {
        uint32_t value = (uint32_t)round(decoded[i]);
        for (int j = 0; j < 4 && output.size() < size; j++) {
            output.push_back((uint8_t)((value >> (j * 8)) & 0xFF));
        }
    }
    
    // Verify integrity
    bool match = (compressed == output);
    cout << "Data integrity: " << (match ? "✓ PERFECT" : "⚠ CHECK") << endl;
    
    if (!match) {
        int errors = 0;
        for (size_t i = 0; i < min(compressed.size(), output.size()); i++) {
            if (compressed[i] != output[i]) errors++;
        }
        cout << "Mismatches: " << errors << " / " << size << endl;
    }
    
    // Save
    string output_file = string(argv[1]);
    size_t pos = output_file.find("_compressed");
    if (pos != string::npos) {
        output_file.replace(pos, 11, "_encrypted");
    }
    
    ofstream out(output_file, ios::binary);
    out.write((char*)output.data(), output.size());
    out.close();
    
    cout << "Saved: " << output_file << endl;
    cout << "Total time: " << (enc_time + dec_time) << " ms" << endl;
    
    return 0;
}