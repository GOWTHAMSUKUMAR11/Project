import hashlib
from PIL import Image
import textwrap
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import serialization
import numpy as np
import cv2
import time


def imageEncryptAndDecrypt():
    # Open the image file

    image_file = Image.open("./static/uploads/bwimages.jpg")

    additional_time = 0

    # Convert the image to bytes
    start_time = time.time()

    image_bytes = image_file.tobytes()

    # Generate the hash object using SHA-512 algorithm
    hash_object = hashlib.sha512()

    # Update the hash object with the image bytes
    hash_object.update(image_bytes)

    # Generate the hash value in hexadecimal format
    hash_hex = hash_object.hexdigest()

    # Print the hash value
    # print("SHA-512 hash value:", hash_hex)

    prekeys=textwrap.wrap(hash_hex, 32)
    
    keys=[]
    for i in prekeys:
        dec_value = int(i, 16)
        # print(dec_value)
        keys.append(dec_value%0.9999)
    
    # Key generation
    extra_start = time.time()

    private_key_person1 = ec.generate_private_key(ec.SECP256R1())
    private_key_person2 = ec.generate_private_key(ec.SECP256R1())

    # Key exchange
    public_key_person1 = private_key_person1.public_key()
    public_key_person2 = private_key_person2.public_key()

    #this is used in encryption
    shared_secret_person1 = private_key_person1.exchange(ec.ECDH(), public_key_person2)

    #this is used in decryption
    shared_secret_person2 = private_key_person2.exchange(ec.ECDH(), public_key_person1)

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    key_hash1 = hashlib.sha256(shared_secret_person1).digest()
    key_hash2 = hashlib.sha256(shared_secret_person2).digest()

    # Encrypting keys
    en=[]
    for i in keys:
        original_value = i
        value_bytes = bytes(str(original_value), 'utf-8')
        encrypted_value = bytes([value_bytes[i] ^ key_hash1[i % len(key_hash1)] for i in range(len(value_bytes))])
        en.append(encrypted_value)
    # Decrypt the encrypted value
    # decrypted_value_bytes = bytes([encrypted_value[i] ^ key_hash2[i % len(key_hash2)] for i in range(len(encrypted_value))])
    # decrypted_value = float(decrypted_value_bytes.decode('utf-8'))
    # Output the results
    # print(en)
    # print(encrypted_value)
    # print(f"Original value: {original_value}")
    # print(f"Encrypted value: {encrypted_value}")
    # print(f"decrypted value: {decrypted_value}")

    

    extra_start  = time.time()

    image=cv2.imread('./static/uploads/bwimages.jpg')

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    height,width,c=image.shape

    def logistic_map(x, r):
        key_stream = []
        for i in range(height):
            x = r * x * (1 - x)
            key_stream.append(int(np.round(x * height - 1))%height)
        return key_stream
    r = 3.99

    # Generate the key stream using the logistic map
    key_stream1 = logistic_map(keys[0], r)
    key_stream2 = logistic_map(keys[1], r)
    key_stream3 = logistic_map(keys[2], r)
    key_stream4 = logistic_map(keys[3], r)

    # Print the first 10 values of the key stream1
    # print(key_stream1)

    extra_start  = time.time()

    image = Image.open('./static/uploads/bwimages.jpg')

    image_array = np.array(image)

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    #GENERATING PERMUTATION SEQUENCE

    key = np.array(key_stream1[:5]) # replace with your float array key
    num_rows = image_array.shape[0]
    permutation_indices = np.random.RandomState(seed=key).permutation(num_rows)

    permuted_image_array = image_array[permutation_indices, :]

    extra_start  = time.time()

    permuted_image = Image.fromarray(permuted_image_array)
    permuted_image.save('./static/uploads/1_permuted_image.png')

    
    image = Image.open('./static/uploads/1_permuted_image.png')
    image_array = np.array(image)

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    #GENERATING PERMUTATION SEQUENCE

    key = np.array(key_stream2[:5]) # replace with your float array key
    num_cols = image_array.shape[1]
    # print(num_cols)
    permutation_indices = np.random.RandomState(seed=key).permutation(num_cols)


    ##COLUMN PERMUTATION

    permuted_image_array = image_array[:,permutation_indices]


    extra_start = time.time()

    permuted_image = Image.fromarray(permuted_image_array)
    permuted_image.save('./static/uploads/2_col_permuted_image.png')

    #DCT


    # Load image
    img = cv2.imread('./static/uploads/2_col_permuted_image.png')

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    # Split image into channels
    b, g, r = cv2.split(img)

    # Define DCT function
    def dct2(block):
        return cv2.dct(cv2.dct(block.T).T)



    # Perform DCT on each channel
    dct_b = dct2(b.astype(np.float32))
    dct_g = dct2(g.astype(np.float32))
    dct_r = dct2(r.astype(np.float32))

    dct_img = cv2.merge((dct_b,dct_g,dct_r))

    extra_start = time.time()

    cv2.imwrite('./static/uploads/3_dct_image.png', dct_img)

    


    #GENERATING SUBSTITUTION BOX


    # Load the image data from a file
    image = np.array(Image.open("./static/uploads/3_dct_image.png"))

    extra_end  = time.time()

    additional_time += extra_end - extra_start



    # Calculate the size of the substitution box needed
    sbox_size = image.shape[0] * image.shape[1] * image.shape[2]

    # Generate a random substitution box with the required size
    sbox = np.arange(sbox_size, dtype=np.uint8)
    np.random.shuffle(sbox)

    # Reshape the substitution box to match the input image shape
    sbox = sbox.reshape(image.shape)

    image_array = np.array(sbox)
    key3 = np.array(key_stream3[:5])
    num_rows = image_array.shape[0]

    permutation_indices1 = np.random.RandomState(seed=key3).permutation(num_rows)
    sbox_row_permuted = image_array[permutation_indices1, :]


    image_array = sbox_row_permuted
    key4 = np.array(key_stream4[:5])
    num_columns = image_array.shape[1]
    permutation_indices2 = np.random.RandomState(seed=key4).permutation(num_columns)
    sbox_permuted = image_array[:,permutation_indices2, :]

    #PERFORMING BITWISE XOR
    
    extra_start = time.time()

    imgforps = Image.open("./static/uploads/3_dct_image.png")

    extra_end  = time.time()

    additional_time += extra_end - extra_start

    img_array = np.array(imgforps)
    encrypted_array = np.bitwise_xor(img_array, sbox)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time - additional_time

    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    encrypted_img = Image.fromarray(encrypted_array)

    encrypted_img.save("./static/uploads/4_final_encrypted_img.png")


    #DECRYPTING KEYS USING ELLIPTIC CURVE CRYPTOGRAPHY
    en_len = len(en)
    dkeys=[]
    print(en_len)
    for i in range(en_len):
        a=en[i]
        decrypted_value_bytes = bytes([a[i] ^ key_hash2[i % len(key_hash2)] for i in range(len(a))])
        decrypted_value = float(decrypted_value_bytes.decode('utf-8'))
        dkeys.append(decrypted_value)
    # Output the results
    # print(f"decrypted value: {decrypted_value}")
    # print(dkeys)


    d_image=cv2.imread('./static/uploads/4_final_encrypted_img.png')

    d_flat_image = d_image.flatten()
    d_height,d_width,d_c=d_image.shape

    def d_logistic_map(x, r):
        """
        Generate key stream values using the logistic map.
        """
        d_key_stream = []
        for i in range(d_height):
            x = r * x * (1 - x)
            d_key_stream.append(int(np.round(x * d_height - 1))%d_height)
        return d_key_stream

    # Set the initial value and parameters for the logistic map

    r = 3.99

    # Generate the key stream using the logistic map
    d_key_stream1 = d_logistic_map(dkeys[0], r)
    d_key_stream2 = d_logistic_map(dkeys[1], r)
    d_key_stream3 = d_logistic_map(dkeys[2], r)
    d_key_stream4 = d_logistic_map(dkeys[3], r)


    dkey3 = np.array(d_key_stream3[:5]) # replace with your float array key
    num_rows1 = sbox_permuted.shape[0]
    re_permutation_indices3 = np.random.RandomState(seed=dkey3).permutation(num_rows1)

    dkey4 = np.array(d_key_stream4[:5]) # replace with your float array key
    num_cols1 = sbox_permuted.shape[1]
    re_permutation_indices4 = np.random.RandomState(seed=dkey4).permutation(num_cols1)

        
    inverse_indices3 = np.argsort(re_permutation_indices3)
    inverse_indices4 = np.argsort(re_permutation_indices4)

    sbox_decrept = sbox_permuted[:,inverse_indices4, :]
    sbox_dpermuted = sbox_decrept[inverse_indices3, :]
    # print(sbox_dpermuted)

    #pixel substitution on decrypted image
    deimgforps = Image.open("./static/uploads/4_final_encrypted_img.png")
    deimg_array = np.array(deimgforps)
    decrypted_array = np.bitwise_xor(deimg_array, sbox_dpermuted)
    decrypted_img = Image.fromarray(decrypted_array)
    decrypted_img.save("./static/uploads/5_decrypted_image_from_ps.png")

    #IDCT on decyrpted image grom pixel substitution

    # Load image
    img = cv2.imread('./static/uploads/5_decrypted_image_from_ps.png')

    # Split image into channels
    b, g, r = cv2.split(img)

    # Define IDCT function
    def idct2(block):
        return cv2.idct(cv2.idct(block.T).T)

    idct_b = idct2(dct_b).astype(np.uint8)
    idct_g = idct2(dct_g).astype(np.uint8)
    idct_r = idct2(dct_r).astype(np.uint8)

    idct_img = cv2.merge((idct_b, idct_g, idct_r))
    cv2.imwrite('./static/uploads/6_idct_image.png', idct_img)


    dp_image = Image.open('./static/uploads/6_idct_image.png')
    dp_image_array = np.array(dp_image)

    dkey = np.array(d_key_stream2[:5]) # replace with your float array key
    num_cols1 = dp_image_array.shape[1]
    re_permutation_indices = np.random.RandomState(seed=dkey).permutation(num_cols1)

    inverse_indices = np.argsort(re_permutation_indices)

    re_permuted_image_array = dp_image_array[:,inverse_indices, :]

    re_permuted_image = Image.fromarray(re_permuted_image_array)
    re_permuted_image.save('./static/uploads/7_reverse_col_permuted_image.png')

    dp_image = Image.open('./static/uploads/7_reverse_col_permuted_image.png')
    dp_image_array = np.array(dp_image)

    dkey1 = np.array(d_key_stream1[:5]) # replace with your float array key
    num_rows1 = dp_image_array.shape[0]
    re_permutation_indices = np.random.RandomState(seed=dkey1).permutation(num_rows1)


    inverse_indices = np.argsort(re_permutation_indices)

    re_permuted_image_array = dp_image_array[inverse_indices, :]

    re_permuted_image = Image.fromarray(re_permuted_image_array)
    re_permuted_image.save('./static/uploads/8_final_decrypted_image.png')



