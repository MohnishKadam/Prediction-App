class Env {
  static const String baseUrl = String.fromEnvironment('API_BASE_URL',
      defaultValue: 'http://192.168.1.7:5000/');
}
