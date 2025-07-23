import 'package:flutter/foundation.dart';

class CryptoProvider with ChangeNotifier {
  // TODO: Add cryptocurrency data and methods

  bool _isLoading = false;
  bool get isLoading => _isLoading;

  void setLoading(bool value) {
    _isLoading = value;
    notifyListeners();
  }
}
