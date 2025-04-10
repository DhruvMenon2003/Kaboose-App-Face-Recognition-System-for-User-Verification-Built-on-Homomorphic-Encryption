# Kaboose Verification App
https://kabooseverification.global.ssl.fastly.net/ 
A facial recognition verification application for mobile websites that uses homomorphic encryption for secure user verification.

## Features

- **Image Upload**: Users can upload their image for verification
- **Facial Recognition**: Uses DeepFace AI for facial recognition
- **Homomorphic Encryption**: Securely encrypts facial embeddings using CKKS algorithm
- **Real-time Verification**: Stream verification using webcam
- **Secure Storage**: All images are encrypted and securely stored

## How It Works

1. **Image Upload**: When a user uploads their image, it's processed by DeepFace to extract facial embeddings.

2. **Homomorphic Encryption**: The embeddings are encrypted using the CKKS algorithm, allowing computations on the encrypted data without decryption.

3. **Verification**: The system compares the encrypted embeddings with previously stored embeddings to ensure the user doesn't have multiple accounts.

4. **Stream Verification**: The webcam captures the user's face in real-time and verifies it against their uploaded image.

## Security Features

- All facial embeddings are homomorphically encrypted
- Images are deleted after processing
- Encryption keys are securely managed
- No plaintext sensitive data is stored

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Facial Recognition**: DeepFace
- **Encryption**: TenSEAL (CKKS homomorphic encryption)
