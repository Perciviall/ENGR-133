import matplotlib.pyplot as plt  # Importing matplotlib for image handling
import numpy as np  # Importing numpy for numerical operations
from demo_encrypt import xor  # Importing the xor function from the demo_encrypt module


def decrypt_vigenere(plaintext, key):
    """Decrypts a given plaintext using the Vigenère cipher with the provided key."""
    decrypted_message = ""  # Initialize the decrypted message as an empty string
    key = key.upper()  # Normalize the key to uppercase for consistent encryption/decryption
    key_index = 0  # Initialize the key index for tracking the position in the key

    for char in plaintext:  # Iterate through each character in the plaintext
        if char.isalpha():  # Check if the character is an alphabetic letter
            # Calculate the shift based on the corresponding key character
            shift = ord(key[key_index % len(key)]) - ord('A')  
            if char.islower():  # If the character is lowercase
                # Perform decryption for lowercase letters
                decrypted_message += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:  # If the character is uppercase
                # Perform decryption for uppercase letters
                decrypted_message += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            key_index += 1  # Increment the key index
        elif char.isdigit():  # Check if the character is a digit
            # Shift the digit according to the key and add to decrypted message
            shift = ord(key[key_index % len(key)]) - ord('A')  
            decrypted_message += str((int(char) - shift) % 10)  # Decrypt the digit
            key_index += 1  # Increment the key index
        elif char.isspace():  # If the character is a space
            decrypted_message += char  # Add space directly to the decrypted message
            key_index += 1  # Increment the key index, as space takes an encoding slot
        else:
            decrypted_message += char  # Non-alphabetic characters are added unchanged

    print(decrypted_message)  # Print the decrypted message for verification
    return decrypted_message  # Return the decrypted message


def decrypt_xor(binary, key):
    """Decrypts a binary string using XOR encryption with the specified key."""
    plaintext = binary_to_plaintext(binary)  # Convert binary string to plaintext
    original = xor(plaintext, key)  # Use XOR function to decrypt the plaintext with the key
    
    return binary_to_plaintext(original)  # Convert the decrypted binary back to plaintext


def decrypt_caesar(plaintext, key):
    """Decrypts a given plaintext using the Caesar cipher with the specified key."""
    new_text = ''  # Initialize the decrypted text as an empty string
    alphabet = 'abcdefghijklmnopqrstuvwxyz'  # Lowercase alphabet
    Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Uppercase alphabet
    number = '0123456789'  # String of digits
    key_alpha = int(key) % 26  # Normalize the key for alphabetic shifting
    key_num = int(key) % 10  # Normalize the key for numeric shifting
    
    for char in plaintext:  # Iterate through each character in the plaintext
        if char.isalpha():  # Check if the character is an alphabetic letter
            if char.isupper():  # If the character is uppercase
                location = (Alphabet.find(char) - key_alpha) % 26  # Calculate new character location
                new_text += Alphabet[location]  # Add the decrypted character to the new text
            else:  # If the character is lowercase
                location = (alphabet.find(char) - key_alpha) % 26  # Calculate new character location
                new_text += alphabet[location]  # Add the decrypted character to the new text
        elif char.isdigit():  # Check if the character is a digit
            location = (number.find(char) - key_num) % 10  # Calculate new digit location
            new_text += number[location]  # Add the decrypted digit to the new text
        else: 
            new_text += char  # Non-alphabetic characters are added unchanged
    
    print(f'Decrypted Message using Caesar Cipher: {new_text}')  # Print the decrypted message
    return new_text  # Return the decrypted message


def convert_to_binary(seq):
    """Converts a text sequence to a binary string."""
    return ''.join(format(ord(c), '08b') for c in seq)  # Format each character as 8-bit binary


def convert_number_to_binary(seq):
    """Converts a sequence of numbers to its binary representation."""
    binary = ""  # Initialize binary string as empty
    # Convert each character in the sequence (assumed to be numbers) to its ASCII character
    for x in range(len(seq)):
        ascii_char = seq[x]  # Get the ASCII character for the current position
        # Convert the ASCII character to its binary representation, formatted as 8 bits
        binary += format(ord(ascii_char), '08b')
    return binary  # Return the complete binary representation


def binary_to_plaintext(binary_string):
    """Converts a binary string back to plaintext."""
    # Remove any whitespace to handle continuous binary strings
    binary_string = binary_string.replace(' ', '')
    
    # Ensure the binary string length is a multiple of 8
    if len(binary_string) % 8 != 0:
         binary_string = binary_string.ljust(len(binary_string) + (8 - len(binary_string) % 8), '0')

    plaintext = ""  # Initialize plaintext as an empty string
    
    for i in range(0, len(binary_string), 8):  # Process each 8-bit chunk
        bv = binary_string[i:i+8]  # Get the current 8-bit chunk
        
        # Convert each binary chunk to its corresponding character
        decimal_value = int(bv, 2)  # Convert binary to decimal
        if decimal_value > 255:  # Check if the decimal value exceeds byte range
            raise OverflowError(f"Binary value '{bv}' exceeds maximum byte value of 255.")
        
        plaintext += chr(decimal_value)  # Convert decimal to character and add to plaintext
    
    return plaintext  # Return the final plaintext


def extract_lsb_from_image(image_path):
    """Extracts the Least Significant Bits (LSB) from the given image."""
    img = plt.imread(image_path)  # Read the image file
    
    binary_data = ""  # Initialize binary data as an empty string

    if len(img.shape) == 2:  # If the image is grayscale
        for row in img:  # Iterate through each row of the image
            for pixel in row:  # Iterate through each pixel in the row
                pixel_value = int(pixel * 255)  # Convert to 0-255 range if it's normalized
                binary_data += format(pixel_value, '08b')[-1]  # Extract LSB and add to binary data
    else:  # If the image is a color image
        for row in img:  # Iterate through each row of the image
            for pixel in row:  # Iterate through each pixel in the row
                r, g, b = [int(channel * 255) for channel in pixel[:3]]  # Convert channels to 0-255
                binary_data += format(r, '08b')[-1]  # Extract LSB from red channel
                binary_data += format(g, '08b')[-1]  # Extract LSB from green channel
                binary_data += format(b, '08b')[-1]  # Extract LSB from blue channel

    return binary_data  # Return the accumulated binary data


def extract_message(binary_data, start_seq, end_seq):
    """Extracts the hidden message located between the specified start and end sequences."""
    start_index = binary_data.find(start_seq)  # Find the start sequence in binary data
    end_index = binary_data.find(end_seq, start_index + len(start_seq))  # Find the end sequence after start

    if start_index == -1 or end_index == -1:  # Check if sequences are found
        return "Start or end sequence not found in the image."  # Return error message

    # Extract the message in between the start and end sequences
    message = binary_data[start_index + len(start_seq):end_index]
    return "Extracted Message: " + message  # Return the extracted message


def main():
    """Main function to execute the decryption process based on user input."""
    decrypt_choice = input("Enter the cipher you want to use for decryption: ")  # Get decryption choice
    key = input("Enter the key for the cipher: ")  # Get the key for decryption
    start_sequence = input("Enter the start sequence: ")  # Get the start sequence for message extraction
    end_sequence = input("Enter the end sequence: ")  # Get the end sequence for message extraction
    input_image = input("Enter the path of the input image: ")  # Get the path to the input image

    start_seq_binary = convert_to_binary(start_sequence)  # Convert start sequence to binary
    end_seq_binary = convert_to_binary(end_sequence)  # Convert end sequence to binary

    binary_data = extract_lsb_from_image(input_image)  # Extract binary data from the image
    
    encoded_message = extract_message(binary_data, start_seq_binary, end_seq_binary)  # Extract the hidden message
    
    # Extract just the binary data from the encoded message
    binary_message = encoded_message.split(": ")[-1].strip()  # Get binary message portion
    
    print(f'Extracted Binary Message: {binary_message}')  # Print the extracted binary message
    print(f'Converted Binary Text: {binary_to_plaintext(binary_message)}')  # Convert binary to plaintext and print

    plaintext = binary_to_plaintext(binary_message)  # Convert binary message to plaintext

    # Decrypt the plaintext based on user-selected cipher type
    if decrypt_choice == 'xor':
        decrypted_plaintext = decrypt_xor(binary_message, key)  # Decrypt using XOR
        print(f'Converted Text: {decrypted_plaintext}')  # Print decrypted text
    elif decrypt_choice == 'caesar':
        decrypted_plaintext = decrypt_caesar(plaintext, key)  # Decrypt using Caesar cipher
        print(f'Converted Text: {decrypted_plaintext}')  # Print decrypted text

    elif decrypt_choice == 'vigenere':
        decrypted_plaintext = decrypt_vigenere(plaintext, key)  # Decrypt using Vigenère cipher
        print(f'Converted Text: {decrypted_plaintext}')  # Print decrypted text


if __name__ == "__main__":
    main()  # Execute the main function when the script is run
