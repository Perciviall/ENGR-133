import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def vigenere(plaintext, key):
    encrypted_message = ""  # Initialize an empty string to store the encrypted message.
    key = key.upper()  # Normalize the key to uppercase for consistent shifting.
    key_index = 0  # Track the position within the key.

    # Iterate over each character in the plaintext.
    for char in plaintext:
        if char.isalpha():  # Check if the character is alphabetic.
            shift = ord(key[key_index % len(key)]) - ord('A')  # Calculate the shift value using the key.
            if char.islower():
                # Encrypt lowercase alphabetic characters.
                encrypted_message += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                # Encrypt uppercase alphabetic characters.
                encrypted_message += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            key_index += 1  # Move to the next position in the key.
        elif char.isdigit():  # Check if the character is a digit.
            shift = ord(key[key_index % len(key)]) - ord('A')  # Use the key to determine shift for digits.
            encrypted_message += str((int(char) + shift) % 10)  # Encrypt digits and wrap around using modulo 10.
            key_index += 1  # Move to the next position in the key.
        else:
            # If the character is a space or other non-alphabetic character, keep it unchanged.
            encrypted_message += char
            if char.isspace():  # Only increment key_index for spaces, not punctuation.
                key_index += 1
    
    # Print the final encrypted message.
    print(f'Encrypted Message using Vigenere Cipher: {encrypted_message}')
    return encrypted_message
def xor(plaintext, key):
    new_key = ''  # Initialize an empty string for the repeated key.
    binary_plain = convert_to_binary(plaintext)  # Convert the plaintext into a binary string.
    cyphertext = ''  # Initialize an empty string for the resulting ciphertext.
    
    # Extend the key to match the length of the plaintext.
    # Commented out loop could have been used to extend the key:
    # while len(new_key) < len(plaintext):
    #     for char in key:
    #         new_key += char
    key_length = len(key)  # Get the length of the original key.
    for i in range(len(plaintext)):
        new_key += key[i % key_length]  # Repeat the key to match the length of the plaintext.

    # Convert the repeated key to binary.
    if new_key != '':
        binary_key = convert_to_binary(new_key)
    else: 
        binary_key = convert_to_binary(key)
    
    # Perform XOR between each bit of the binary plaintext and the binary key.
    for i in range(0, len(binary_plain)):
        if binary_plain[i] == binary_key[i]:
            cyphertext += '0'  # Add '0' if bits are the same.
        else:
            cyphertext += '1'  # Add '1' if bits are different.
    return cyphertext  # Return the resulting binary string as ciphertext.

def caesar(plaintext, key):
    new_text = ''  # Initialize an empty string for the resulting encrypted text.
    alphabet = 'abcdefghijklmnopqrstuvwxyz'  # Define the lowercase alphabet.
    Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Define the uppercase alphabet.
    key = int(key) % 26  # Normalize the key within the range of 0-25.

    # Iterate over each character in the plaintext.
    for char in plaintext:
        if char.isalpha():  # Check if the character is alphabetic.
            if char.isupper():
                location = (Alphabet.find(char) + key) % 26  # Find new position for uppercase characters.
                new_text += Alphabet[location]  # Append the shifted character to the result.
            else:
                location = (alphabet.find(char) + key) % 26  # Find new position for lowercase characters.
                new_text += alphabet[location]  # Append the shifted character to the result.
        elif char.isdigit():
            return 'Please enter all letters'  # Return error message if input contains digits.
        else: 
            new_text += char  # Preserve spaces and other non-alphabetic characters.
    
    # Print the final encrypted message.
    print(f'Encrypted Message using Caesar Cipher: {new_text}')
    return new_text  # Return the resulting encrypted text.

def convert_to_binary(seq):
    """Convert text sequence to binary."""
    return ''.join(format(ord(c), '08b') for c in seq)  # Convert each character to its 8-bit binary representation.

def convert_number_to_binary(seq):
    binary = ""  # Initialize an empty string to store the binary representation.
    
    # Convert the sequence (number) to its ASCII character
    for x in range(len(seq)):
        ascii_char = seq[x]  # Extract each character from the sequence.
        
        # Convert the ASCII character to its binary representation, formatted as 8 bits.
        binary += format(ord(ascii_char), '08b')
    
    return binary  # Return the full binary representation as a string.

def binary_to_plaintext(binary_string):
    # Remove any whitespace to handle continuous binary strings.
    binary_string = binary_string.replace(' ', '')
    
    # Ensure the binary string length is a multiple of 8.
    if len(binary_string) % 8 != 0:
        raise ValueError("Binary string length must be a multiple of 8.")  # Raise an error if length is invalid.
    
    # Split the binary string into chunks of 8 bits.
    plaintext = ""  # Initialize an empty string to store the decoded plaintext.
    
    # Iterate over the binary string in chunks of 8 bits.
    for i in range(0, len(binary_string), 8):
        bv = binary_string[i:i+8]  # Get 8-bit chunk.
        
        # Convert each binary chunk to its corresponding character.
        decimal_value = int(bv, 2)  # Convert the binary value to a decimal integer.
        if decimal_value > 255:
            raise OverflowError(f"Binary value '{bv}' exceeds maximum byte value of 255.")  # Raise an error if value is too large.
        
        plaintext += chr(decimal_value)  # Append the corresponding character to the plaintext.
    
    return plaintext  # Return the decoded plaintext as a string.

def encode_image(input_image, binary_message, offset, encoded_image_path):
    """Embed the encoded message into the image by changing the Least Significant Bit (LSB) of each pixel."""
    img = plt.imread(input_image)  # Read the input image.
    
    # Check if the binary message fits within the image's pixel capacity.
    if(len(binary_message) > img.size):
        print("Given message is too long to be encoded in the image.")
    else:
        counter = 0  # Counter for iterating over the binary message.
        offsetCounter = 0  # Counter to manage the offset for encoding.
        binary_message = binary_message.replace(" ", "")  # Remove spaces from the binary message.
        
        if len(img.shape) == 2:  # Check if the image is grayscale.
            rows, cols = img.shape  # Get the dimensions of the grayscale image.
                
            # Iterate over each pixel in the grayscale image.
            for row in range(rows):
                for col in range(cols):
                    if counter < len(binary_message):
                        if (offsetCounter >= offset):  # Apply the offset before modifying the pixel.
                            pixel = img[row, col]  # Get the current pixel value.
                            # Convert pixel to binary and modify the LSB.
                            binary_pixel = format(int(pixel * 255), '08b')
                            new_pixel = binary_pixel[-1] + binary_message[counter]  # Replace LSB.
                            img[row, col] = int(new_pixel, 2) / 255  # Set the new pixel value.
                            counter += 1  # Move to the next bit of the binary message.
                            offsetCounter += 1
                        else:
                            break  # Stop if offsetCounter hasn't reached the required offset.
                            # offsetCounter += 1  # Uncomment to increase offset counter if needed.
                offsetCounter = 0  # Reset the offset counter after each row.
            
            # Convert the modified image back to PIL format and save it.
            pil_image = Image.fromarray((img * 255).astype(np.uint8), mode="L")
            pil_image.save(encoded_image_path)
            plt.show()
        
        else:  # If the image is in color (RGB).
            img = img[:,:,:3]  # Ensure only RGB channels are processed.
            rows, cols, _ = img.shape  # Get dimensions of the color image.
            
            # Iterate over each pixel in the RGB image.
            for row in range(rows):
                for col in range(cols):
                    if counter < len(binary_message):
                        if(offsetCounter >= offset):  # Apply the offset before modifying the pixel.
                            pixel = img[row, col]  # Get the current pixel (RGB).
                            r, g, b = [int(channel * 255) for channel in pixel[:3]]  # Convert channels to 0-255 range.
                            
                            # Modify the LSB for red, green, and blue channels.
                            binary_r = format(r, '08b')
                            binary_g = format(g, '08b')
                            binary_b = format(b, '08b')
                            
                            # Embed message bit into the red channel.
                            if counter < len(binary_message):  
                                new_r = binary_r[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_r = binary_r
                                
                            # Embed message bit into the green channel.
                            if counter < len(binary_message):  
                                new_g = binary_g[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_g = binary_g
                                
                            # Embed message bit into the blue channel.
                            if counter < len(binary_message):  
                                new_b = binary_b[:-1] + binary_message[counter]
                                counter += 1
                            else:
                                new_b = binary_b
                            
                            # Assign the modified RGB values back to the image.
                            img[row, col] = [int(new_r, 2) / 255, int(new_g, 2) / 255, int(new_b, 2) / 255]
                        else:
                            offsetCounter += 1  # Increment offset counter until offset is reached.
                offsetCounter = 0  # Reset the offset counter after each row.
            
            # Convert the modified image back to PIL format and save it.
            pil_image = Image.fromarray((img * 255).astype(np.uint8), mode="RGB")
            pil_image.save(encoded_image_path)
        
        print(f"Message successfully encoded and saved to: {encoded_image_path}")  # Confirm successful encoding.

def image_compare(input_image, encode_image_path, image_compare_path):

    # Remove unnecessary dimensions from the input and encoded images.
    input_image_data = plt.imread(input_image)
    encoded_image_data = plt.imread(encode_image_path)

    print(input_image_data - encoded_image_data)  # Print the difference between the images for debugging.

    # Check if images are in RGBA format (3-dimensional data).
    if(np.ndim(input_image_data) == 3 and np.ndim(encoded_image_data) == 3):
        input_image_data = input_image_data[:,:,:3]  # Keep only RGB channels.
        encoded_image_data = encoded_image_data[:,:,:3]  # Keep only RGB channels.
    
    # Check if one image is grayscale while the other is RGB.
    if(np.ndim(input_image_data) == 2 and np.ndim(encoded_image_data) == 3):
        # Convert encoded image to grayscale if it contains "gry" or "gray" in the filename.
        if(encode_image_path.index("gry") > -1 or encode_image_path.index("gray") > -1):
            encoded_image_data = encoded_image_data[:,:,0]  # Use the first channel.
    
    # Create a new array for storing comparison results.
    newArray = np.zeros_like(input_image_data, dtype=np.uint8)

    # Check if both arrays have the same shape and size.
    if np.shape(input_image_data) == np.shape(encoded_image_data) and np.size(input_image_data) == np.size(encoded_image_data):
        # Check if the arrays are 2-dimensional (grayscale).
        if(np.ndim(input_image_data) == 2):
            # Iterate over each pixel in the grayscale images.
            for i in range(len(input_image_data)):
                for j in range(len(input_image_data[0])):
                    value = input_image_data[i][j]
                    # If pixels match, set value to 0 (black); otherwise, set to 255 (white).
                    if value == encoded_image_data[i][j]:
                        newArray[i][j] = 0 
                    else:
                        newArray[i][j] = 255

            # Save the comparison image as 'white_and_black.png' with a grayscale colormap.
            path = 'white_and_black.png'
            plt.imsave(path, newArray, cmap='gray')
        
        # Treat the arrays as 3-dimensional (RGB).
        else:
            # Iterate over each pixel and each RGB channel.
            for i in range(len(input_image_data)):
                for j in range(len(input_image_data[0])):
                    for k in range(len(input_image_data[0][0])):
                        value = input_image_data[i][j][k]
                        # If channel values match, set value to 0 (black); otherwise, set to 255 (white).
                        if value == encoded_image_data[i][j][k]:
                            newArray[i][j][k] = 0
                        else: 
                            newArray[i][j][k] = 255
            
            # Convert the comparison array to an RGB image and save as 'white_and_black.png'.
            pil_image = Image.fromarray(newArray, mode="RGB")
            path = 'white_and_black.png'
            pil_image.save(path)
    
    # Check if the input and encoded images are identical.
    if(np.array_equal(input_image_data, encoded_image_data)):
        print("The images are the same.")
        return True  # Return True if images are identical.
    else:
        print("The images are different.")
        return False  # Return False if images differ.
    
def add_space_between_8_bits(binary_string):
    # Split the binary string into chunks of 8 bits and join them with a space.
    spaced_binary = ' '.join(binary_string[i:i+8] for i in range(0, len(binary_string), 8))
    return spaced_binary  # Return the spaced-out binary string.


def main():
    # Prompt the user to choose a cipher for encryption.
    cipher_choice = input("Enter the cipher you want to use for encryption: ")
    # Prompt the user to enter the plaintext to encrypt.
    plaintext = input("Enter the plaintext you want to encrypt: ")
    # Prompt the user to enter the key for the selected cipher.
    key = input("Enter the key for the cipher: ")
    # Prompt the user for the start and end sequences for the binary message.
    start_sequence = (input("Enter the start sequence: "))
    end_sequence = (input("Enter the end sequence: "))
    # Prompt the user for the bit offset for encoding.
    bit_offset = int(input("Enter the bit offset before you want to start encoding: "))
    # Prompt the user for the path of the input image.
    input_image = input("Enter the path of the input image: ")
    # Prompt the user for the path to save the encoded image.
    encoded_image_path = input("Enter the path for your encoded image: ")
    # Prompt the user for the path of the image to compare against.
    image_compare_path = input("Enter the path of the image you want to compare: ")

    # Convert the start and end sequences to binary format.
    binary_start_sequence = convert_number_to_binary(start_sequence)
    binary_end_sequence = convert_number_to_binary(end_sequence)
    
    # Process the encryption based on the selected cipher.
    if cipher_choice == 'xor':
        # Encrypt the plaintext using the XOR cipher.
        encoded_binary_message = xor(plaintext, key)
        # Combine the binary start sequence, encrypted message, and end sequence.
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        # Format the combined message with spaces between 8-bit segments.
        combined_message_spaced = add_space_between_8_bits(combined_message)
        # Print the encrypted message and its binary representation.
        print(f'Encrypted Message using XOR Cipher: {binary_to_plaintext(encoded_binary_message)}')
        print(f"Binary Output Message: {combined_message_spaced}")
    elif cipher_choice == 'caesar':
        # Encrypt the plaintext using the Caesar cipher.
        encoded__message = caesar(plaintext, key)
        # Convert the encoded message to binary format.
        encoded_binary_message = convert_to_binary(encoded__message)
        # Combine the binary start sequence, encoded message, and end sequence.
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        # Format the combined message with spaces.
        combined_message_spaced = add_space_between_8_bits(combined_message)
        print(f"Binary Output Message: {combined_message_spaced}")
    elif cipher_choice == 'vigenere':
        # Encrypt the plaintext using the Vigenère cipher.
        encoded__message = vigenere(plaintext, key)
        # Convert the encoded message to binary format.
        encoded_binary_message = convert_to_binary(encoded__message)
        # Combine the binary start sequence, encoded message, and end sequence.
        combined_message = binary_start_sequence + encoded_binary_message + binary_end_sequence
        # Format the combined message with spaces.
        combined_message_spaced = add_space_between_8_bits(combined_message)
        print(f"Binary Output Message: {combined_message_spaced}")
    else:
        # Handle invalid cipher choices.
        print("Error: Please enter a valid cipher >:(")

    # Encode the binary message into the image and save it.
    encode_image(input_image, combined_message_spaced, bit_offset, encoded_image_path)
    # Compare the original and encoded images.
    image_compare(input_image, encoded_image_path, image_compare_path)

# Execute the main function if this script is run directly.
if __name__ == "__main__":
    main()
