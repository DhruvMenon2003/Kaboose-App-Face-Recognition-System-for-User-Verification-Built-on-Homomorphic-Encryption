# Kaboose Verification App

A facial recognition verification application for mobile websites that uses homomorphic encryption for secure user verification.

## Features

- **Image Upload**: Users can upload their image for verification
- **Facial Recognition**: Uses DeepFace AI for facial recognition
- **Homomorphic Encryption**: Securely encrypts facial embeddings using CKKS algorithm
- **Real-time Verification**: Stream verification using webcam
- **Secure Storage**: All images are encrypted and securely stored

## Setup Instructions

### Prerequisites

- Python 3.7+ installed
- Node.js and npm (optional, for development)

### Local Installation

1. Clone or download this repository

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the Python backend server:
   ```
   python verify.py
   ```

4. In a separate terminal, start the frontend server:
   ```
   python -m http.server 8000
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Deployment

For detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

### Quick Deployment

#### Backend Deployment (Heroku)

1. Make sure you have the Heroku CLI installed
2. Run the deployment script:
   ```
   bash deploy-backend.sh your-app-name
   ```
3. Note the URL of your deployed backend

#### Frontend Deployment (Netlify)

1. Make sure you have the Netlify CLI installed
2. Run the deployment script with your backend URL:
   ```
   node deploy-frontend.js https://your-backend-url.com
   ```

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