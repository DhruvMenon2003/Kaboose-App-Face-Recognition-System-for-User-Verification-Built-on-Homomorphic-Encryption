# Kaboose Verification App - Backend for Facial Recognition

import os
import base64
import math
import json
import random
import logging
import threading
import time
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kaboose-verification')

# Try to import the required libraries, with fallback instructions if not available
try:
    import tenseal as ts
    from deepface import DeepFace
    import cv2
    from PIL import Image
except ImportError as e:
    logger.error(f"Required library not found: {str(e)}")
    logger.error("Please install all required libraries using: pip install tenseal deepface flask flask-cors opencv-python pillow")
    exit(1)

app = Flask(__name__)

# Configure CORS - in production, this should be restricted to your frontend domain
# For local development, allow all origins
if os.environ.get('FLASK_ENV') == 'production':
    # In production, restrict CORS to specific origins
    # Replace with your actual frontend domain when deployed
    CORS(app, origins=['https://your-frontend-domain.com'])
else:
    # In development, allow all origins
    CORS(app)

# Directory to store encrypted embeddings
EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Constants for encryption and security
POLY_MODULUS_DEGREE = 8192  # Higher degree = more secure but slower
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]  # Bit sizes for coefficient modulus
GLOBAL_SCALE = 2**40  # Scale for CKKS encoding
NOISE_VARIANCE = 0.01  # Variance for noise addition during encryption
DISTANCE_THRESHOLD = 0.8  # Threshold for facial similarity (lower = more strict)

# Helper functions for data handling
def write_data(file_name: str, data: bytes) -> None:
    """Write data to a file with base64 encoding if needed"""
    if isinstance(data, bytes):
        # bytes to base64
        data = base64.b64encode(data)
    
    with open(file_name, 'wb') as f:
        f.write(data)
    logger.info(f"Data written to {file_name}")

def read_data(file_name: str) -> bytes:
    """Read data from a file and decode from base64"""
    with open(file_name, "rb") as f:
        data = f.read()
    
    # base64 to bytes
    return base64.b64decode(data)

def add_noise_to_vector(vector: List[float], variance: float = NOISE_VARIANCE) -> List[float]:
    """Add Gaussian noise to a vector for additional security"""
    return [val + random.gauss(0, variance) for val in vector]

def normalize_vector(vector: List[float]) -> List[float]:
    """Normalize a vector to unit length"""
    norm = math.sqrt(sum(x*x for x in vector))
    if norm == 0:
        return vector
    return [x/norm for x in vector]

# Initialize encryption context
def initialize_encryption() -> str:
    """Initialize the CKKS encryption scheme for homomorphic operations"""
    try:
        # Create context for CKKS scheme with higher security parameters
        context = ts.context(ts.SCHEME_TYPE.CKKS, 
                           poly_modulus_degree=POLY_MODULUS_DEGREE, 
                           coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
        
        # Generate Galois keys for homomorphic operations
        context.generate_galois_keys()
        context.global_scale = GLOBAL_SCALE
        
        # Save secret and public contexts
        secret_context = context.serialize(save_secret_key=True)
        write_data(os.path.join(EMBEDDINGS_DIR, "secret.txt"), secret_context)
        
        # Create public context for sharing
        context.make_context_public()
        public_context = context.serialize()
        write_data(os.path.join(EMBEDDINGS_DIR, "public.txt"), public_context)
        
        logger.info("Encryption initialized successfully with CKKS scheme")
        return "Encryption initialized successfully"
    except Exception as e:
        logger.error(f"Error initializing encryption: {str(e)}")
        raise

# Compute Euclidean distance between encrypted vectors
def compute_encrypted_distance(enc_vec1: ts.CKKSVector, enc_vec2: ts.CKKSVector) -> ts.CKKSVector:
    """Compute Euclidean distance between two encrypted vectors
    
    This function computes the distance without decrypting the vectors,
    leveraging homomorphic properties of the CKKS scheme.
    """
    try:
        # Calculate (vec1 - vec2)^2 homomorphically
        diff = enc_vec1 - enc_vec2
        squared_diff = diff.square()
        
        # Sum the squared differences (dot product with ones vector)
        # This is equivalent to the squared Euclidean distance
        return squared_diff.sum()
    except Exception as e:
        logger.error(f"Error computing encrypted distance: {str(e)}")
        raise

# API endpoint for image verification
@app.route('/api/verify', methods=['POST'])
def verify():
    """Verify a user's image against existing encrypted embeddings"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Save the uploaded image temporarily
    image_file = request.files['image']
    image_path = os.path.join(EMBEDDINGS_DIR, 'temp_image.jpg')
    image_file.save(image_path)
    
    try:
        logger.info("Processing uploaded image for verification")
        
        # Get face embedding using DeepFace
        embedding_result = DeepFace.represent(img_path=image_path, model_name="Facenet")
        
        # Extract the embedding vector
        if isinstance(embedding_result, list) and len(embedding_result) > 0:
            embedding = embedding_result[0]['embedding']
        else:
            embedding = embedding_result['embedding']
            
        # Normalize the embedding vector
        embedding = normalize_vector(embedding)
        
        # Add noise to the embedding for additional security
        noisy_embedding = add_noise_to_vector(embedding)
        
        # Check if we need to initialize encryption
        if not os.path.exists(os.path.join(EMBEDDINGS_DIR, "secret.txt")):
            initialize_encryption()
        
        # Load encryption context
        context = ts.context_from(read_data(os.path.join(EMBEDDINGS_DIR, "secret.txt")))
        
        # Encrypt the embedding
        enc_embedding = ts.ckks_vector(context, noisy_embedding)
        
        # Save original image for stream verification
        original_image_path = os.path.join(EMBEDDINGS_DIR, 'latest_original.jpg')
        image_file.seek(0)  # Reset file pointer
        image_file.save(original_image_path)
        
        # Generate a unique user ID
        user_id = f"user_{len([f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith('user_')])}"
        
        # Compare with existing embeddings
        existing_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("user_") and f.endswith(".txt")]
        
        # If there are existing embeddings, compare with them
        if existing_files:
            logger.info(f"Comparing with {len(existing_files)} existing embeddings")
            
            for file in existing_files:
                try:
                    # Load existing embedding
                    existing_enc = ts.ckks_vector_from(context, read_data(os.path.join(EMBEDDINGS_DIR, file)))
                    
                    # Calculate Euclidean distance homomorphically
                    distance_enc = compute_encrypted_distance(enc_embedding, existing_enc)
                    
                    # Decrypt the distance for comparison
                    distance = math.sqrt(distance_enc.decrypt()[0])
                    
                    logger.info(f"Distance from {file}: {distance}")
                    
                    # If distance is less than threshold, user already exists
                    if distance < DISTANCE_THRESHOLD:
                        logger.warning(f"Duplicate user detected with distance {distance}")
                        # Delete the temporary image
                        if os.path.exists(image_path):
                            os.remove(image_path)
                        return jsonify({
                            'verified': False, 
                            'message': 'User already exists',
                            'distance': distance
                        }), 200
                except Exception as e:
                    logger.error(f"Error comparing with {file}: {str(e)}")
                    continue
        
        # Save the encrypted embedding
        write_data(os.path.join(EMBEDDINGS_DIR, f"{user_id}.txt"), enc_embedding.serialize())
        
        # Save the original image for stream verification
        os.rename(original_image_path, os.path.join(EMBEDDINGS_DIR, f"{user_id}_original.jpg"))
        
        logger.info(f"New user {user_id} verified and saved successfully")
        
        return jsonify({
            'verified': True, 
            'message': 'Image verified successfully',
            'user_id': user_id
        }), 200
    
    except Exception as e:
        logger.error(f"Error during verification: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temporary image
        if os.path.exists(image_path):
            os.remove(image_path)

# API endpoint for stream verification
# Constants for video streaming
STREAM_FPS = 10
STREAM_QUALITY = 0.8

# Custom exceptions for camera operations
class CannotOpenCamera(Exception):
    """Exception raised if the camera cannot be opened."""

class CannotReadCamera(Exception):
    """Exception raised if the camera cannot be read."""

# Server-side video stream handler
class ServerVideoStream:
    def __init__(self, fps=STREAM_FPS):
        self.fps = fps
        self.lock = threading.Lock()
        self.frame = None
        self.active = False
        self.thread = None
        self.cameras = {}
        self.camera_index = 0
        self._stop_thread = False

    def start_stream(self, camera_index=0):
        """Start the video stream with the specified camera"""
        if self.active:
            return
        
        self.camera_index = camera_index
        self.get_camera(camera_index)
        
        self.active = True
        self._stop_thread = False
        self.thread = threading.Thread(target=self._update_frames)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started video stream from camera {camera_index}")

    def stop_stream(self):
        """Stop the video stream and release resources"""
        self._stop_thread = True
        self.active = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish
            
        # Release all cameras
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()
        logger.info("Stopped video stream and released camera resources")

    def get_camera(self, index):
        """Get or initialize a camera by index"""
        if index in self.cameras:
            return self.cameras[index]

        # Initialize new camera
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise CannotOpenCamera(f"Cannot open camera {index}")

        self.cameras[index] = cap
        logger.info(f"Initialized camera {index}")
        return cap

    def _update_frames(self):
        """Background thread to continuously update frames"""
        while not self._stop_thread and self.active:
            start_time = time.time()
            
            try:
                camera = self.get_camera(self.camera_index)
                ret, frame = camera.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_index}")
                    time.sleep(0.1)  # Short delay before retry
                    continue
                    
                with self.lock:
                    self.frame = frame
                    
            except Exception as e:
                logger.error(f"Error capturing frame: {str(e)}")
                time.sleep(0.5)  # Longer delay on error
                continue
                
            # Maintain consistent frame rate
            if self.fps > 0:
                interval = 1.0 / self.fps
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

    def get_frame(self):
        """Get the current frame safely"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()
            
    def get_jpeg_frame(self, quality=STREAM_QUALITY):
        """Get current frame as JPEG bytes with base64 encoding"""
        frame = self.get_frame()
        if frame is None:
            return None
            
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality * 100)])
        frame_bytes = buffer.tobytes()
        return base64.b64encode(frame_bytes).decode('utf-8')
        
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop_stream()

# Initialize server video stream handler
video_stream = ServerVideoStream()

# API endpoint for video stream verification
@app.route('/api/stream', methods=['POST'])
def stream():
    """Handle continuous video stream verification against stored user images"""
    try:
        # Get the user ID from the request
        user_id = request.form.get('user_id', 'latest')
        
        # If user_id is 'latest', get the most recent user
        if user_id == 'latest':
            existing_files = [f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("user_") and f.endswith("_original.jpg")]
            if not existing_files:
                return jsonify({'error': 'No registered users found'}), 400
            user_id = sorted(existing_files)[-1].split('_original')[0]
        
        # Check if the original image exists
        original_image_path = os.path.join(EMBEDDINGS_DIR, f"{user_id}_original.jpg")
        if not os.path.exists(original_image_path):
            return jsonify({'error': f'Original image for {user_id} not found'}), 404

        # Start video stream if not already running
        try:
            if not video_stream.active:
                # Default to camera 0, but allow specifying a different camera
                camera_index = int(request.form.get('camera_index', 0))
                video_stream.start_stream(camera_index=camera_index)
        except CannotOpenCamera as e:
            return jsonify({'error': f'Could not start camera: {str(e)}'}), 500

        # Get current frame with timeout/retry
        max_retries = 3
        frame = None
        
        for attempt in range(max_retries):
            frame = video_stream.get_frame()
            if frame is not None:
                break
            time.sleep(0.5)  # Wait briefly before retry
        
        if frame is None:
            return jsonify({'error': 'Could not capture frame after multiple attempts'}), 500

        # Save frame temporarily
        temp_path = os.path.join(EMBEDDINGS_DIR, 'temp_stream.jpg')
        cv2.imwrite(temp_path, frame)

        try:
            # Verify the stream frame against the stored original image
            result = DeepFace.verify(
                img1_path=temp_path,
                img2_path=original_image_path,
                model_name="Facenet",
                distance_metric="euclidean"
            )

            # Get frame as base64 encoded JPEG
            frame_b64 = video_stream.get_jpeg_frame(quality=STREAM_QUALITY)
            
            # Return verification result with frame
            return jsonify({
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'frame': frame_b64,
                'user_id': user_id
            }), 200

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"Error during stream verification: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API endpoint to stop the video stream
@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the video stream and release camera resources"""
    try:
        # Stop the video stream
        video_stream.stop_stream()
        
        # Clean up any temporary files
        temp_path = os.path.join(EMBEDDINGS_DIR, 'temp_stream.jpg')
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({'message': 'Stream stopped successfully'}), 200
    except Exception as e:
        logger.error(f"Error stopping stream: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API endpoint to get encryption status
@app.route('/api/encryption/status', methods=['GET'])
def encryption_status():
    """Check if encryption is initialized"""
    is_initialized = os.path.exists(os.path.join(EMBEDDINGS_DIR, "secret.txt"))
    return jsonify({
        'initialized': is_initialized,
        'algorithm': 'CKKS',
        'poly_modulus_degree': POLY_MODULUS_DEGREE,
        'security_level': 'high'
    })

# API endpoint to initialize encryption
@app.route('/api/encryption/initialize', methods=['POST'])
def init_encryption():
    """Initialize or reinitialize encryption"""
    try:
        result = initialize_encryption()
        return jsonify({'message': result}), 200
    except Exception as e:
        logger.error(f"Error initializing encryption: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start the server when run directly
if __name__ == '__main__':
    # Initialize encryption on startup if not already initialized
    if not os.path.exists(os.path.join(EMBEDDINGS_DIR, "secret.txt")):
        try:
            initialize_encryption()
        except Exception as e:
            logger.error(f"Error initializing encryption on startup: {str(e)}")
    
    # Get port from environment variable (for Heroku compatibility)
    port = int(os.environ.get('PORT', 5000))
    
    # Set debug mode based on environment
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    logger.info(f"Starting server on port {port} with debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)