import 'package:flutter/material.dart';
import 'screens/crypto_dashboard.dart';

void main() {
  runApp(const CryptoDashboardApp());
}

class CryptoDashboardApp extends StatelessWidget {
  const CryptoDashboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CoinGecko Crypto Dashboard',
      theme: ThemeData(
        primaryColor: const Color(0xFF667eea),
        visualDensity: VisualDensity.adaptivePlatformDensity,
        fontFamily: 'Segoe UI',
      ),
      home: const CryptoDashboard(),
      debugShowCheckedModeBanner: false,
    );
  }
}
