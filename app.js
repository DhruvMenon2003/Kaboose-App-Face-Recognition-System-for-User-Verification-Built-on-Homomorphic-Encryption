/*
  This is your site JavaScript code!
*/

// Print a message in the browser's dev tools console each time the page loads
// Use your menus or right-click / control-click and choose "Inspect" > "Console"
console.log("Hello 🌎");

// ----- KABOOSE VERIFICATION APP CODE -----

// Kaboose Verification App - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Get all screens
    const initialScreen = document.getElementById('initial-screen');
    const imageUploadScreen = document.getElementById('image-upload-screen');
    const videoCaptureScreen = document.getElementById('video-capture-screen');
    const verificationFailedScreen = document.getElementById('verification-failed-screen');
    const catVerificationScreen = document.getElementById('cat-verification-screen');
    
    // Get loading video element
    const loadingVideo = document.getElementById('loading-video');

    // API endpoints
    // This allows the API URL to be overridden by an environment variable or use the default for local development
    // Modified to work with Glitch's URL structure
    const API_BASE_URL = window.API_URL || window.location.origin + '/api';
    let currentUserId = null;
    let videoStream = null;
    let videoElement = null;
    let currentCameraFacingMode = 'environment'; // Default to rear camera

    // Hide all screens except the initial one
    function showScreen(screen) {
        // Hide all screens
        const screens = document.querySelectorAll('.screen');
        screens.forEach(s => s.style.display = 'none');
        
        // Show the selected screen
        screen.style.display = 'flex';
    }

    // Function to capture image from video stream
    function captureImage(videoEl) {
        const canvas = document.createElement('canvas');
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
        return new Promise((resolve) => {
            canvas.toBlob(resolve, 'image/jpeg');
        });
    }

    // Function to stop video stream
    function stopVideoStream() {
        if (videoStream) {
            videoStream.getTracks().forEach(track => track.stop());
            videoStream = null;
        }
    }

    // Function to reset video UI
    function resetVideoUI() {
        stopVideoStream();
        const videoPlaceholder = document.querySelector('#video-capture-screen .video-placeholder');
        videoPlaceholder.innerHTML = '<img src="assets/video-icon.svg" alt="Video Icon" id="video-icon">';
        videoElement = null;
        
        // Re-attach event listener to the new video icon
        document.getElementById('video-icon').addEventListener('click', startVideoCapture);
    }

    // Function to start camera with specified facing mode
    async function startCamera(facingMode) {
        try {
            const constraints = {
                video: {
                    facingMode: { exact: facingMode }
                }
            };
            
            try {
                videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                currentCameraFacingMode = facingMode;
            } catch (error) {
                // Fallback to any available camera if specified camera is not available
                console.log(`${facingMode} camera not available, falling back to default camera:`, error);
                videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
                currentCameraFacingMode = 'user'; // Assume default is front camera
            }
            
            videoElement.srcObject = videoStream;
        } catch (error) {
            console.error('Error accessing camera:', error);
            showScreen(verificationFailedScreen);
        }
    }

    // Function to switch camera
    async function switchCamera() {
        stopVideoStream();
        const newFacingMode = currentCameraFacingMode === 'environment' ? 'user' : 'environment';
        await startCamera(newFacingMode);
    }

    // Function to handle video capture
    async function startVideoCapture() {
        try {
            // Create video element if it doesn't exist
            if (!videoElement) {
                videoElement = document.createElement('video');
                videoElement.autoplay = true;
                videoElement.playsInline = true;
                videoElement.style.width = '100%';
                videoElement.style.height = '100%';
                videoElement.style.objectFit = 'cover';
                
                // Create camera switch button
                const switchButton = document.createElement('button');
                switchButton.textContent = '📷 Switch Camera';
                switchButton.className = 'camera-switch-btn';
                switchButton.addEventListener('click', function(e) {
                    e.stopPropagation();
                    switchCamera();
                });
                
                // Create status indicator
                const statusIndicator = document.createElement('div');
                statusIndicator.className = 'verification-status';
                statusIndicator.textContent = 'Ready';
                statusIndicator.style.position = 'absolute';
                statusIndicator.style.bottom = '10px';
                statusIndicator.style.left = '10px';
                statusIndicator.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
                statusIndicator.style.color = 'white';
                statusIndicator.style.padding = '5px 10px';
                statusIndicator.style.borderRadius = '4px';
                statusIndicator.style.fontSize = '14px';
                
                // Replace the video icon with the video element and add buttons
                const videoPlaceholder = document.querySelector('#video-capture-screen .video-placeholder');
                videoPlaceholder.innerHTML = '';
                videoPlaceholder.appendChild(videoElement);
                videoPlaceholder.appendChild(switchButton);
                videoPlaceholder.appendChild(statusIndicator);
            }
            
            // Start camera with environment (rear) facing mode by default
            await startCamera('environment');
            
            // Update status to indicate verification is starting
            const statusIndicator = document.querySelector('.verification-status');
            if (statusIndicator) {
                statusIndicator.textContent = 'Starting verification...';
            }
            
            // Wait a moment for the user to see themselves before starting verification
            setTimeout(async () => {
                try {
                    // Update status
                    if (statusIndicator) {
                        statusIndicator.textContent = 'Verifying...';
                    }
                    
                    // Capture image from video without interrupting the stream
                    const imageBlob = await captureImage(videoElement);
                    
                    // Create form data for API request
                    const formData = new FormData();
                    formData.append('image', imageBlob);
                    formData.append('user_id', currentUserId || 'latest');
                    
                    // Call the stream API in the background
                    const response = await fetch(`${API_BASE_URL}/stream`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    // Update status based on verification result
                    if (statusIndicator) {
                        statusIndicator.textContent = result.verified ? 'Verification successful!' : 'Verification failed';
                        statusIndicator.style.backgroundColor = result.verified ? 'rgba(0, 128, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
                    }
                    
                    // Wait a moment to show the status before transitioning
                    setTimeout(() => {
                        // Stop the video stream and reset UI
                        resetVideoUI();
                        
                        if (result.verified) {
                            // If verification successful, show success screen
                            showScreen(catVerificationScreen);
                        } else {
                            // If verification failed, show failed screen
                            showScreen(verificationFailedScreen);
                        }
                    }, 1500); // Show status for 1.5 seconds before transitioning
                    
                } catch (error) {
                    console.error('Error during stream verification:', error);
                    if (statusIndicator) {
                        statusIndicator.textContent = 'Error during verification';
                        statusIndicator.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
                    }
                    
                    // Wait a moment before showing error screen
                    setTimeout(() => {
                        resetVideoUI();
                        showScreen(verificationFailedScreen);
                    }, 1500);
                }
            }, 1000); // Reduced from 3 seconds to 1 second for better UX
        } catch (error) {
            console.error('Error accessing camera:', error);
            showScreen(verificationFailedScreen);
        }
    }

    // Listen for the video to end and then transition to the image upload screen
    loadingVideo.addEventListener('ended', function() {
        showScreen(imageUploadScreen);
    });
    
    // Fallback: If video doesn't play or ends too quickly, transition after a delay
    setTimeout(() => {
        if (initialScreen.style.display !== 'none') {
            showScreen(imageUploadScreen);
        }
    }, 5000);

    // Image upload functionality
    const imageIcon = document.getElementById('image-icon');
    imageIcon.addEventListener('click', function() {
        // Create a file input element
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);
        
        // Trigger click on the file input
        fileInput.click();
        
        // Handle file selection
        fileInput.addEventListener('change', async function() {
            if (fileInput.files && fileInput.files[0]) {
                // Show loading screen while processing
                showScreen(initialScreen);
                
                try {
                    // Create form data for API request
                    const formData = new FormData();
                    formData.append('image', fileInput.files[0]);
                    
                    // Call the verify API
                    const response = await fetch(`${API_BASE_URL}/verify`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.verified) {
                        // If verification successful, proceed to video capture
                        showScreen(videoCaptureScreen);
                    } else {
                        // If verification failed (user already exists), show failed screen
                        showScreen(verificationFailedScreen);
                    }
                } catch (error) {
                    console.error('Error during verification:', error);
                    showScreen(verificationFailedScreen);
                } finally {
                    // Clean up the file input
                    document.body.removeChild(fileInput);
                }
            }
        });
    });

    // Video capture functionality - attach event listener
    const videoIcon = document.getElementById('video-icon');
    videoIcon.addEventListener('click', startVideoCapture);

    // Back button functionality
    const backIcon = document.getElementById('back-icon');
    backIcon.addEventListener('click', function() {
        // Reset video UI
        resetVideoUI();
        showScreen(imageUploadScreen);
    });

    // Retry button functionality
    const retryIcon = document.getElementById('retry-icon');
    retryIcon.addEventListener('click', function() {
        // Show loading screen briefly before retrying
        showScreen(initialScreen);
        setTimeout(() => {
            showScreen(imageUploadScreen);
        }, 1000);
    });

    // For demo purposes, allow clicking on the cat icon to go back to the beginning
    const catIcon = document.getElementById('cat-icon');
    catIcon.addEventListener('click', function() {
        showScreen(initialScreen);
        // Restart the flow after a delay
        setTimeout(() => {
            showScreen(imageUploadScreen);
        }, 2000);
    });
});

// ----- GLITCH STARTER PROJECT HELPER CODE -----

// Only show the fileopening links in editor 
try {
  if (
    window.self === window.top ||
    window.location.ancestorOrigins.length > 1
  ) {
    let fileopeners = Array.from(document.getElementsByClassName("fileopener"));
    fileopeners.forEach((fo) => {
      fo.classList.remove("fileopener");
    });
  }
} catch (e) {}

// Open file when the link in the preview is clicked
let goto = (file, line) => {
  window.parent.postMessage(
    { type: "glitch/go-to-line", payload: { filePath: file, line: line } },
    "*"
  );
};

// Get the file opening button from its class name
const filer = document.querySelectorAll(".fileopener");
filer.forEach((f) => {
  f.onclick = () => {
    goto(f.dataset.file, f.dataset.line);
  };
});