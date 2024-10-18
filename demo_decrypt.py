import matplotlib.pyplot as plt
import numpy as np
from demo_encrypt import xor


def decrypt_vigenere(plaintext, key):
    decrypted_message = ""
    key = key.upper()  # Normalize the key to uppercase
    key_index = 0

    for char in plaintext:
        if char.isalpha():  # Shift alphabetic characters.
            shift = ord(key[key_index % len(key)]) - ord('A')
            if char.islower():
                decrypted_message += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                decrypted_message += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            key_index += 1
        elif char.isdigit():  # Shift numeric characters.
            shift = ord(key[key_index % len(key)]) - ord('A')
            decrypted_message += str((int(char) - shift) % 10)
            key_index += 1
        elif char.isspace():
            decrypted_message += char  # Add space as is.
            key_index += 1  # Space takes up an encoding slot.
        else:
            decrypted_message += char  # Keep other non-alphabetic characters unchanged.
    

    print(decrypted_message)
    return decrypted_message

def decrypt_xor(binary, key):
    plaintext = binary_to_plaintext(binary)
    original = xor(plaintext,key)
    
    return binary_to_plaintext(original)

def decrypt_caesar(plaintext, key):
    new_text = ''
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    number = '0123456789'
    key_alpha = int(key) % 26
    key_num = int(key) % 10
    
    for char in plaintext:
        if char.isalpha():
            if char.isupper():
                location = (Alphabet.find(char) - key_alpha) % 26
                new_text += Alphabet[location]
            else:
                location = (alphabet.find(char) - key_alpha) % 26
                new_text += alphabet[location]
        elif char.isdigit():
            location = (number.find(char) - key_num) % 10
            new_text += number[location]
        else: 
            new_text += char
    
    print(f'Decrypted Message using Caesar Cipher: {new_text}')
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
         binary_string = binary_string.ljust(len(binary_string) + (8 - len(binary_string) % 8), '0')

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


def extract_lsb_from_image(image_path):
    """Extract Least Significant Bits (LSB) from the image."""
    img = plt.imread(image_path)
    
    binary_data = ""

    if len(img.shape) == 2:  # Grayscale image
        for row in img:
            for pixel in row:
                pixel_value = int(pixel * 255)  # Convert to 0-255 range if it's normalized
                binary_data += format(pixel_value, '08b')[-1]
    else:  # Color image
        for row in img:
            for pixel in row:
                r, g, b = [int(channel * 255) for channel in pixel[:3]]  # Convert to 0-255
                binary_data += format(r, '08b')[-1]  # Red channel LSB
                binary_data += format(g, '08b')[-1]  # Green channel LSB
                binary_data += format(b, '08b')[-1]  # Blue channel LSB

    return binary_data

def extract_message(binary_data, start_seq, end_seq):
    """Extract the hidden message between the start and end sequence."""
    start_index = binary_data.find(start_seq)
    end_index = binary_data.find(end_seq, start_index + len(start_seq))

    if start_index == -1 or end_index == -1:
        return "Start or end sequence not found in the image."

    # Extract the message in between
    message = binary_data[start_index + len(start_seq):end_index]
    return "Extracted Message: " + message


def main():
    decrypt_choice = input("Enter the cipher you want to use for decryption: ")
    key = input("Enter the key for the cipher: ")
    start_sequence = input("Enter the start sequence: ")
    end_sequence = input("Enter the end sequence: ")
    input_image = input("Enter the path of the input image: ")

    start_seq_binary = convert_to_binary(start_sequence)
    end_seq_binary = convert_to_binary(end_sequence)

    binary_data = extract_lsb_from_image(input_image)
    
    encoded_message = extract_message(binary_data, start_seq_binary, end_seq_binary)
    
    # Extract just the binary data from the message
    binary_message = encoded_message.split(": ")[-1].strip()
    
    print(f'Extracted Binary Message: {binary_message}')
    print(f'Converted Binary Text: {binary_to_plaintext(binary_message)}')

    plaintext = binary_to_plaintext(binary_message)

    if decrypt_choice == 'xor':
        decrypted_plaintext = decrypt_xor(binary_message, key)
        print(f'Converted Text: {decrypted_plaintext}')
    elif decrypt_choice == 'caesar':
        decrypted_plaintext = decrypt_caesar(plaintext, key)
        print(f'Converted Text: {decrypted_plaintext}')

    elif decrypt_choice == 'vigenere':
        decrypted_plaintext = decrypt_vigenere(plaintext, key)
        print(f'Converted Text: {decrypted_plaintext}')

if __name__ == "__main__":
    main()