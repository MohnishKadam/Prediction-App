class Env {
  // Update this URL to match your Flask backend
  // For local development, use your computer's IP address instead of localhost
  // Example: 'http://192.168.1.100:5000' or 'http://10.0.2.2:5000' for Android emulator
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:5000',
  );
  
  // Alternative URLs for different environments
  static const String localUrl = 'http://localhost:5000';
  static const String androidEmulatorUrl = 'http://10.0.2.2:5000';
  
  // Get the appropriate URL based on the platform
  static String getBaseUrl() {
    // You can add platform-specific logic here
    return baseUrl;
  }
}