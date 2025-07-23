import 'package:flutter/material.dart';

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

class CryptoDashboard extends StatefulWidget {
  const CryptoDashboard({super.key});

  @override
  _CryptoDashboardState createState() => _CryptoDashboardState();
}

class _CryptoDashboardState extends State<CryptoDashboard> {
  int selectedTab = 0;
  final List<String> tabs = [
    'Market',
    'Search',
    'Trending',
    'Global',
    'Wallet',
    'Prediction'
  ];

  // Prediction tab state
  final TextEditingController predictCoinController = TextEditingController();
  String? predictionResult;

  @override
  void dispose() {
    predictCoinController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: SafeArea(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Card(
                elevation: 8,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20)),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    children: [
                      headerSection(),
                      const SizedBox(height: 20),
                      tabBar(),
                      const SizedBox(height: 20),
                      getTabContent(),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget headerSection() {
    return Column(
      children: [
        ShaderMask(
          shaderCallback: (Rect bounds) {
            return const LinearGradient(
              colors: [Color(0xFF667EEA), Color(0xFF764BA2)],
            ).createShader(bounds);
          },
          child: const Text(
            '🚀 CoinGecko Crypto Dashboard',
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
        ),
        const SizedBox(height: 10),
        const Text('Real-time cryptocurrency data at your fingertips'),
        const SizedBox(height: 10),
        const Text('💼 Wallet Balance: \$10000',
            style: TextStyle(fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget tabBar() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceAround,
      children: List.generate(
        tabs.length,
        (index) => Expanded(
          child: GestureDetector(
            onTap: () => setState(() => selectedTab = index),
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 4),
              padding: const EdgeInsets.symmetric(vertical: 12),
              decoration: BoxDecoration(
                color: selectedTab == index
                    ? const Color(0xFF667EEA)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Center(
                child: Text(
                  tabs[index],
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: selectedTab == index ? Colors.white : Colors.grey,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget getTabContent() {
    switch (selectedTab) {
      case 0:
        return marketTab();
      case 1:
        return searchTab();
      case 2:
        return trendingTab();
      case 3:
        return globalTab();
      case 4:
        return walletTab();
      case 5:
        return const Center(child: Text('Prediction tab is not available.'));
      default:
        return Container();
    }
  }

  Widget marketTab() {
    return columnCard('📊 Market Data', [
      Row(
        children: [
          Expanded(
            child: DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: 'Currency'),
              items: ['USD', 'EUR', 'BTC', 'ETH', 'DOGE']
                  .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                  .toList(),
              onChanged: (_) {},
            ),
          ),
          const SizedBox(width: 10),
          const Expanded(
            child: TextField(
              decoration: InputDecoration(labelText: 'Results Per Page'),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(width: 10),
          ElevatedButton(
              onPressed: () {}, child: const Text('Get Market Data')),
        ],
      ),
    ]);
  }

  Widget searchTab() {
    return Column(
      children: [
        columnCard('🔍 Search Cryptocurrencies', [
          Row(
            children: [
              const Expanded(
                child: TextField(
                    decoration: InputDecoration(labelText: 'Enter Coin Name')),
              ),
              const SizedBox(width: 10),
              ElevatedButton(onPressed: () {}, child: const Text('Search')),
            ],
          ),
        ]),
        const SizedBox(height: 10),
        columnCard('💰 Get Specific Coin Data', [
          Row(
            children: [
              const Expanded(
                child: TextField(
                    decoration: InputDecoration(labelText: 'Coin ID')),
              ),
              const SizedBox(width: 10),
              ElevatedButton(
                  onPressed: () {}, child: const Text('Get Details')),
            ],
          ),
        ]),
      ],
    );
  }

  Widget trendingTab() {
    return columnCard('🔥 Trending Cryptocurrencies', [
      ElevatedButton(onPressed: () {}, child: const Text('Get Trending Coins')),
    ]);
  }

  Widget globalTab() {
    return columnCard('🌍 Global Market Statistics', [
      ElevatedButton(onPressed: () {}, child: const Text('Get Global Data')),
    ]);
  }

  Widget walletTab() {
    return columnCard('💼 Wallet Holdings', [
      ElevatedButton(
          onPressed: () {}, child: const Text('Refresh Wallet Info')),
      const SizedBox(height: 20),
      columnCard('💼 Dummy Wallet', [
        Row(
          children: [
            Expanded(
              child: DropdownButtonFormField<String>(
                decoration: const InputDecoration(labelText: 'Select Coin'),
                items: ['Bitcoin', 'Ethereum', 'Cardano']
                    .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                    .toList(),
                onChanged: (_) {},
              ),
            ),
            const SizedBox(width: 10),
            const Expanded(
              child:
                  TextField(decoration: InputDecoration(labelText: 'Quantity')),
            ),
            const SizedBox(width: 10),
            ElevatedButton(onPressed: () {}, child: const Text('Buy')),
            const SizedBox(width: 10),
            ElevatedButton(onPressed: () {}, child: const Text('Sell')),
          ],
        ),
        const SizedBox(height: 10),
        const Text('💰 Balance: \$10000'),
        const Text('📊 Holdings: {}'),
      ]),
    ]);
  }

  Widget columnCard(String title, List<Widget> children) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title,
              style:
                  const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 10),
          ...children,
        ]),
      ),
    );
  }
}
