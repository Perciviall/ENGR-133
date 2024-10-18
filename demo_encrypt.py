import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def vigenere(plaintext, key):
    encrypted_message = ""
    key = key.upper()  # Normalize the key to uppercase
    key_index = 0

    for char in plaintext:
        if char.isalpha():  # Shift alphabetic characters.
            shift = ord(key[key_index % len(key)]) - ord('A')
            if char.islower():
                encrypted_message += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                encrypted_message += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            key_index += 1
        elif char.isdigit():  # Shift numeric characters.
            shift = ord(key[key_index % len(key)]) - ord('A')  # Use key's letter for numeric shift
            encrypted_message += str((int(char) + shift) % 10)
            key_index += 1
        else:  # If it's a space or other non-alphabetic character, add it unchanged.
            encrypted_message += char
            if char.isspace():  # Only increment key_index for spaces, not for punctuation.
                key_index += 1
    print(f'Encrypted Message using Vigenere Cipher: {encrypted_message}')
    return encrypted_message

def xor(plaintext, key):
    new_key = ''
    binary_plain = convert_to_binary(plaintext)
    cyphertext = ''
    
    # while len(new_key) < len(plaintext):
    #     for char in key:
    #         new_key += char
    key_length = len(key)
    for i in range(len(plaintext)):
        new_key += key[i % key_length]

    if new_key != '':
        binary_key = convert_to_binary(new_key)
    else: 
        binary_key = convert_to_binary(key)
    for i in range(0,len(binary_plain)):
        if binary_plain[i] == binary_key[i]:
            cyphertext += '0'
        else:
            cyphertext += '1'
    return cyphertext

def caesar(plaintext,key):
    new_text = ''
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    key = int(key) % 26
    
    for char in plaintext:
        if char.isalpha():
            if char.isupper():
                location = Alphabet.find(char) + key
                location = (Alphabet.find(char) + key) % 26
                new_text += Alphabet[location]
            else:
                location = alphabet.find(char) + key
                location = (alphabet.find(char) + key) % 26
                new_text += alphabet[location]
        elif char.isdigit():
            return 'Please enter all letters'
        else: 
            new_text += char
    print(f'Encrypted Message using Caesar Cipher: {new_text}')
    return new_text

def convert_to_binary(seq):
    """Convert text sequence to binary."""
    return ''.join(format(ord(c), '08b') for c in seq)

def convert_number_to_binary(seq):
    binary = ""
    # Convert the sequence (number) to its ASCII character
    for x in range(len(seq)):
        ascii_char = seq[x]
    # Convert the ASCII character to its binary representation, formatted as 8 bits
        binary += format(ord(ascii_char), '08b')
    return binary

def binary_to_plaintext(binary_string):
    # Remove any whitespace to handle continuous binary strings
    binary_string = binary_string.replace(' ', '')
    
    # Ensure the binary string length is a multiple of 8
    if len(binary_string) % 8 != 0:
        raise ValueError("Binary string length must be a multiple of 8.")
    
    # Split the binary string into chunks of 8 bits
    plaintext = ""
    
    for i in range(0, len(binary_string), 8):
        bv = binary_string[i:i+8]  # Get 8-bit chunk
        
        # Convert each binary chunk to its corresponding character
        decimal_value = int(bv, 2)
        if decimal_value > 255:
            raise OverflowError(f"Binary value '{bv}' exceeds maximum byte value of 255.")
        
        plaintext += chr(decimal_value)
    
    return plaintext

def encode_image(input_image, binary_message, offset, encoded_image_path):
    """Embed the encoded message into the image by changing the Least Significant Bit (LSB) of each pixel."""
    img = plt.imread(input_image)
    
    if(len(binary_message) > img.size):
        print("Given message is too long to be encoded in the image.")
    else:
        counter = 0
        offsetCounter = 0
        binary_message = binary_message.replace(" ", "")
        if len(img.shape) == 2:  # Grayscale image
            rows, cols = img.shape
                
            for row in range(rows):
                for col in range(cols):
                    if counter < len(binary_message):
                        if (offsetCounter >= offset):
                            pixel = img[row, col]
                            # Convert pixel to binary and modify the LSB
                            binary_pixel = format(int(pixel * 255), '08b')
                            new_pixel = binary_pixel[-1] + binary_message[counter]  # Replace LSB
                            img[row, col] = int(new_pixel, 2) / 255  # Set new pixel value
                            counter += 1
                            offsetCounter += 1
                        else:
                            break
                            #offsetCounter += 1
                offsetCounter = 0                
            pil_image = Image.fromarray((img * 255).astype(np.uint8), mode="L")
            pil_image.save(encoded_image_path)
            plt.show()
        else:  # Color image
            img = img[:,:,:3]
            rows, cols, _ = img.shape
            for row in range(rows):
                for col in range(cols):
                    if counter < len(binary_message):
                        if(offsetCounter >= offset):
                            pixel = img[row, col]
                            r, g, b = [int(channel * 255) for channel in pixel[:3]]  # Convert to 0-255
                            # Modify the LSB for red, green, and blue channels
                            binary_r = format(r, '08b')
                            binary_g = format(g, '08b')
                            binary_b = format(b, '08b')
                            
                            if counter < len(binary_message):  # Red channel
                                new_r = binary_r[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_r = binary_r
                                
                            if counter < len(binary_message):  # Green channel
                                new_g = binary_g[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_g = binary_g
                                
                            if counter < len(binary_message):  # Blue channel
                                new_b = binary_b[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_b = binary_b
                            
                            # Assign the modified RGB values back to the image
                            img[row, col] = [int(new_r, 2) / 255, int(new_g, 2) / 255, int(new_b, 2) / 255]
                        else:
                            offsetCounter += 1
                offsetCounter = 0
            pil_image = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")
            pil_image.save(encoded_image_path)
        print(f"Message successfully encoded and saved to: {encoded_image_path}")

def image_compare(input_image, encode_image_path, image_compare_path):

    #remove unecessary dimensions
    input_image_data = plt.imread(input_image)
    encoded_image_data = plt.imread(encode_image_path)

    print(input_image_data-encoded_image_data)

    #check if RGBA nonsense
    if(np.ndim(input_image_data) == 3 and np.ndim(encoded_image_data) == 3):
        input_image_data = input_image_data[:,:,:3]
        encoded_image_data = encoded_image_data[:,:,:3]
    
    if(np.ndim(input_image_data) == 2 and np.ndim(encoded_image_data) == 3):
        if(encode_image_path.index("gry") > -1 or encode_image_path.index("gray") > -1):
            encoded_image_data = encoded_image_data[:,:,0]
    
    #creates new array for storage
    newArray = np.zeros_like(input_image_data, dtype=np.uint8)

    #checks if both arrays are the same shape
    if np.shape(input_image_data) == np.shape(encoded_image_data) and np.size(input_image_data) == np.size(encoded_image_data):
        #checks if the arrays are 2 dimensions
        if(np.ndim(input_image_data) == 2):
            for i in range(len(input_image_data)):
                for j in range(len(input_image_data[0])):
                    value = input_image_data[i][j]
                    if value == encoded_image_data[i][j]:
                        newArray[i][j] = 0 
                    else:
                        newArray[i][j] = 255

            path = 'white_and_black.png'
            plt.imsave(path, newArray, cmap='gray')
        
        #treats the arrays as 3 dimensions
        else:
            for i in range(len(input_image_data)):
                for j in range(len(input_image_data[0])):
                    for k in range(len(input_image_data[0][0])):

                        value = input_image_data[i][j][k]
                        if value == encoded_image_data[i][j][k]:
                            newArray[i][j][k] = 0
                        else: 
                            newArray[i][j][k] = 255
            
            pil_image = Image.fromarray(newArray, mode="RGB")
            path = 'white_and_black.png'
            pil_image.save(path)
    
    if(np.array_equal(input_image_data, encoded_image_data)):
        print("The images are the same.")
        return True
    else:
        print("The images are different.")
        return False
    
def add_space_between_8_bits(binary_string):
    # Split the binary string into chunks of 8 bits
    spaced_binary = ' '.join(binary_string[i:i+8] for i in range(0, len(binary_string), 8))
    return spaced_binary

def main():
    cipher_choice = input("Enter the cipher you want to use for encryption: ")
    plaintext = input("Enter the plaintext you want to encrypt: ")
    key = input("Enter the key for the cipher: ")
    start_sequence = (input("Enter the start sequence: "))
    end_sequence = (input("Enter the end sequence: "))
    bit_offset = int(input("Enter the bit offset before you want to start encoding: "))
    input_image = input("Enter the path of the input image: ")
    encoded_image_path = input("Enter the path for your encoded image: ")
    image_compare_path = input("Enter the path of the image you want to compare: ")

    binary_start_sequence = convert_number_to_binary(start_sequence)
    binary_end_sequence = convert_number_to_binary(end_sequence)
    
    if cipher_choice == 'xor':
        encoded_binary_message = xor(plaintext, key)
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        combined_message_spaced = add_space_between_8_bits(combined_message)
        print(f'Encrypted Message using XOR Cipher: {binary_to_plaintext(encoded_binary_message)}')
        print(f"Binary Output Message: {combined_message_spaced}")
    elif cipher_choice == 'caesar':
        encoded__message = caesar(plaintext, key)
        encoded_binary_message = convert_to_binary(encoded__message)
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        combined_message_spaced = add_space_between_8_bits(combined_message)
        print(f"Binary Output Message: {combined_message_spaced}")
    elif cipher_choice == 'vigenere':
        encoded__message = vigenere(plaintext, key)
        encoded_binary_message = convert_to_binary(encoded__message)
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        combined_message_spaced = add_space_between_8_bits(combined_message)
        print(f"Binary Output Message: {combined_message_spaced}")
    else:
        print("Error: Please enter a valid cipher >:(")

    encode_image(input_image, combined_message_spaced, bit_offset, encoded_image_path)
    image_compare(input_image, encoded_image_path, image_compare_path)

if __name__ == "__main__":
    main()