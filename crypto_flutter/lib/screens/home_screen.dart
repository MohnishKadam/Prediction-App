import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/crypto_provider.dart';

// Define the CryptoDashboard class if it doesn't exist
class CryptoDashboard extends StatelessWidget {
  const CryptoDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crypto Dashboard'),
      ),
      body: const Center(
        child: Text('Welcome to Crypto Dashboard!'),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<CryptoProvider>(
      builder: (context, provider, child) {
        return const CryptoDashboard();
      },
    );
  }
}
