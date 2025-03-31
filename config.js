// Kaboose Verification App - Configuration

// This file contains configuration settings for the application
// It allows switching between development and production environments

// Set the API base URL based on the current environment
window.API_URL = (() => {
    // Check if we're in a production environment (based on hostname)
    const isProduction = 
        window.location.hostname !== 'localhost' && 
        window.location.hostname !== '127.0.0.1';
    
    // Return the appropriate API URL
    if (isProduction) {
        // Replace this URL with your actual deployed backend URL
        return 'https://your-backend-url.com/api';
    } else {
        // Local development URL
        return 'http://localhost:5000/api';
    }
})();

console.log('API URL configured:', window.API_URL);