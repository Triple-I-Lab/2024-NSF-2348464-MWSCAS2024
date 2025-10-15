#include <iostream>
#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

int main() {
    cout << "Testing Phantom-FHE CKKS..." << endl;
    
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));
    
    PhantomContext context(parms);
    cout << "Context created!" << endl;
    
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    cout << "Keys generated!" << endl;
    
    PhantomCKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Slot count: " << slot_count << endl;
    
    vector<double> input(slot_count, 3.14);
    double scale = pow(2.0, 40);
    
    PhantomPlaintext plain;
    encoder.encode(context, input, scale, plain);
    cout << "Encoded!" << endl;
    
    PhantomCiphertext cipher;
    public_key.encrypt_asymmetric(context, plain, cipher);
    cout << "Encrypted!" << endl;
    
    PhantomPlaintext decrypted;
    secret_key.decrypt(context, cipher, decrypted);
    
    vector<double> output;
    encoder.decode(context, decrypted, output);
    cout << "Result: " << output[0] << " (expected 3.14)" << endl;
    cout << "Phantom-FHE works!" << endl;
    
    return 0;
}
