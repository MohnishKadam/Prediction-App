import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../models/wallet_model.dart';
import '../models/prediction_model.dart';
import '../models/market_coin_model.dart';
import '../models/search_coin_model.dart';
import '../models/trending_coin_model.dart';
import '../models/global_stats_model.dart';
import 'package:intl/intl.dart';

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

  // State variables
  WalletModel? wallet;
  List<MarketCoinModel> marketCoins = [];
  List<SearchCoinModel> searchResults = [];
  List<TrendingCoinModel> trendingCoins = [];
  GlobalStatsModel? globalStats;
  bool isLoading = false;
  String? errorMessage;

  // Controllers
  final TextEditingController searchController = TextEditingController();
  final TextEditingController coinIdController = TextEditingController();
  final TextEditingController predictCoinController = TextEditingController();
  final TextEditingController walletAmountController = TextEditingController();
  
  String selectedCurrency = 'usd';
  int perPage = 10;
  String selectedWalletCoin = 'bitcoin';
  PredictionModel? lastPrediction;

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  @override
  void dispose() {
    searchController.dispose();
    coinIdController.dispose();
    predictCoinController.dispose();
    walletAmountController.dispose();
    super.dispose();
  }

  Future<void> _loadInitialData() async {
    await _fetchMarketData();
    await _fetchWallet();
  }

  Future<void> _fetchMarketData() async {
    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final coins = await ApiService.fetchMarketData(
        currency: selectedCurrency,
        perPage: perPage,
      );
      setState(() {
        marketCoins = coins;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
    }
  }

  Future<void> _fetchWallet() async {
    try {
      final walletData = await ApiService.getWallet();
      setState(() {
        wallet = walletData;
      });
    } catch (e) {
      print('Error fetching wallet: $e');
    }
  }

  Future<void> _searchCoins() async {
    if (searchController.text.trim().isEmpty) return;

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final results = await ApiService.searchCoins(searchController.text.trim());
      setState(() {
        searchResults = results;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
    }
  }

  Future<void> _fetchTrending() async {
    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final trending = await ApiService.fetchTrendingCoins();
      setState(() {
        trendingCoins = trending;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
    }
  }

  Future<void> _fetchGlobalStats() async {
    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final stats = await ApiService.fetchGlobalStats();
      setState(() {
        globalStats = stats;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
    }
  }

  Future<void> _predictPrice() async {
    if (predictCoinController.text.trim().isEmpty) return;

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final prediction = await ApiService.fetchAndPredict(predictCoinController.text.trim());
      setState(() {
        lastPrediction = prediction;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
    }
  }

  Future<void> _buyCoin() async {
    final amount = double.tryParse(walletAmountController.text);
    if (amount == null || amount <= 0) {
      _showSnackBar('Please enter a valid amount');
      return;
    }

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final updatedWallet = await ApiService.buy(selectedWalletCoin, amount);
      setState(() {
        wallet = updatedWallet;
        isLoading = false;
      });
      walletAmountController.clear();
      _showSnackBar('Successfully bought $amount $selectedWalletCoin');
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
      _showSnackBar('Buy failed: $e');
    }
  }

  Future<void> _sellCoin() async {
    final amount = double.tryParse(walletAmountController.text);
    if (amount == null || amount <= 0) {
      _showSnackBar('Please enter a valid amount');
      return;
    }

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      final updatedWallet = await ApiService.sell(selectedWalletCoin, amount);
      setState(() {
        wallet = updatedWallet;
        isLoading = false;
      });
      walletAmountController.clear();
      _showSnackBar('Successfully sold $amount $selectedWalletCoin');
    } catch (e) {
      setState(() {
        errorMessage = e.toString();
        isLoading = false;
      });
      _showSnackBar('Sell failed: $e');
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
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
                      _buildHeader(),
                      const SizedBox(height: 20),
                      _buildTabBar(),
                      const SizedBox(height: 20),
                      if (isLoading) _buildLoadingIndicator(),
                      if (errorMessage != null) _buildErrorMessage(),
                      _buildTabContent(),
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

  Widget _buildHeader() {
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
        const Text('Real-time cryptocurrency data with AI predictions'),
        const SizedBox(height: 10),
        Text(
          '💼 Wallet Balance: \$${wallet?.balance.toStringAsFixed(2) ?? "Loading..."}',
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
      ],
    );
  }

  Widget _buildTabBar() {
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
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLoadingIndicator() {
    return const Padding(
      padding: EdgeInsets.all(20.0),
      child: CircularProgressIndicator(),
    );
  }

  Widget _buildErrorMessage() {
    return Container(
      margin: const EdgeInsets.all(10),
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: Colors.red.shade100,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.red),
      ),
      child: Text(
        errorMessage!,
        style: const TextStyle(color: Colors.red),
      ),
    );
  }

  Widget _buildTabContent() {
    switch (selectedTab) {
      case 0:
        return _buildMarketTab();
      case 1:
        return _buildSearchTab();
      case 2:
        return _buildTrendingTab();
      case 3:
        return _buildGlobalTab();
      case 4:
        return _buildWalletTab();
      case 5:
        return _buildPredictionTab();
      default:
        return Container();
    }
  }

  Widget _buildMarketTab() {
    return Column(
      children: [
        _buildCard('📊 Market Data', [
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: selectedCurrency,
                  decoration: const InputDecoration(labelText: 'Currency'),
                  items: ['usd', 'eur', 'btc', 'eth']
                      .map((e) => DropdownMenuItem(value: e, child: Text(e.toUpperCase())))
                      .toList(),
                  onChanged: (value) {
                    setState(() {
                      selectedCurrency = value!;
                    });
                  },
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextFormField(
                  decoration: const InputDecoration(labelText: 'Results Per Page'),
                  keyboardType: TextInputType.number,
                  initialValue: perPage.toString(),
                  onChanged: (value) {
                    perPage = int.tryParse(value) ?? 10;
                  },
                ),
              ),
              const SizedBox(width: 10),
              ElevatedButton(
                onPressed: _fetchMarketData,
                child: const Text('Refresh'),
              ),
            ],
          ),
        ]),
        const SizedBox(height: 20),
        ...marketCoins.map((coin) => _buildCoinCard(coin)).toList(),
      ],
    );
  }

  Widget _buildSearchTab() {
    return Column(
      children: [
        _buildCard('🔍 Search Cryptocurrencies', [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: searchController,
                  decoration: const InputDecoration(labelText: 'Enter Coin Name'),
                ),
              ),
              const SizedBox(width: 10),
              ElevatedButton(
                onPressed: _searchCoins,
                child: const Text('Search'),
              ),
            ],
          ),
        ]),
        const SizedBox(height: 20),
        ...searchResults.map((coin) => _buildSearchResultCard(coin)).toList(),
      ],
    );
  }

  Widget _buildTrendingTab() {
    return Column(
      children: [
        _buildCard('🔥 Trending Cryptocurrencies', [
          ElevatedButton(
            onPressed: _fetchTrending,
            child: const Text('Get Trending Coins'),
          ),
        ]),
        const SizedBox(height: 20),
        ...trendingCoins.map((coin) => _buildTrendingCard(coin)).toList(),
      ],
    );
  }

  Widget _buildGlobalTab() {
    return Column(
      children: [
        _buildCard('🌍 Global Market Statistics', [
          ElevatedButton(
            onPressed: _fetchGlobalStats,
            child: const Text('Get Global Data'),
          ),
        ]),
        const SizedBox(height: 20),
        if (globalStats != null) _buildGlobalStatsCard(),
      ],
    );
  }

  Widget _buildWalletTab() {
    return Column(
      children: [
        _buildCard('💼 Wallet Holdings', [
          ElevatedButton(
            onPressed: _fetchWallet,
            child: const Text('Refresh Wallet Info'),
          ),
        ]),
        const SizedBox(height: 20),
        if (wallet != null) _buildWalletHoldingsCard(),
        const SizedBox(height: 20),
        _buildCard('💼 Trading', [
          Row(
            children: [
              Expanded(
                child: DropdownButtonFormField<String>(
                  value: selectedWalletCoin,
                  decoration: const InputDecoration(labelText: 'Select Coin'),
                  items: ['bitcoin', 'ethereum', 'cardano', 'ripple', 'solana']
                      .map((e) => DropdownMenuItem(value: e, child: Text(e.toUpperCase())))
                      .toList(),
                  onChanged: (value) {
                    setState(() {
                      selectedWalletCoin = value!;
                    });
                  },
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: TextField(
                  controller: walletAmountController,
                  decoration: const InputDecoration(labelText: 'Quantity'),
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: _buyCoin,
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                  child: const Text('Buy'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ElevatedButton(
                  onPressed: _sellCoin,
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                  child: const Text('Sell'),
                ),
              ),
            ],
          ),
        ]),
      ],
    );
  }

  Widget _buildPredictionTab() {
    return Column(
      children: [
        _buildCard('🤖 AI Price Prediction', [
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: predictCoinController,
                  decoration: const InputDecoration(
                    labelText: 'Enter Coin ID (e.g., bitcoin)',
                    hintText: 'bitcoin, ethereum, cardano...',
                  ),
                ),
              ),
              const SizedBox(width: 10),
              ElevatedButton(
                onPressed: _predictPrice,
                child: const Text('Predict'),
              ),
            ],
          ),
        ]),
        const SizedBox(height: 20),
        if (lastPrediction != null) _buildPredictionResultCard(),
      ],
    );
  }

  Widget _buildCard(String title, List<Widget> children) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 10),
            ...children,
          ],
        ),
      ),
    );
  }

  Widget _buildCoinCard(MarketCoinModel coin) {
    final formatter = NumberFormat.currency(symbol: '\$');
    final isPositive = coin.priceChangePercentage24h >= 0;
    
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: ListTile(
        leading: coin.image.isNotEmpty
            ? Image.network(coin.image, width: 40, height: 40)
            : const Icon(Icons.currency_bitcoin),
        title: Text('${coin.name} (${coin.symbol.toUpperCase()})'),
        subtitle: Text(formatter.format(coin.currentPrice)),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              '${isPositive ? '+' : ''}${coin.priceChangePercentage24h.toStringAsFixed(2)}%',
              style: TextStyle(
                color: isPositive ? Colors.green : Colors.red,
                fontWeight: FontWeight.bold,
              ),
            ),
            Text(
              'Vol: ${formatter.format(coin.totalVolume)}',
              style: const TextStyle(fontSize: 10),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchResultCard(SearchCoinModel coin) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: ListTile(
        leading: coin.image.isNotEmpty
            ? Image.network(coin.image, width: 40, height: 40)
            : const Icon(Icons.currency_bitcoin),
        title: Text('${coin.name} (${coin.symbol.toUpperCase()})'),
        subtitle: Text('ID: ${coin.id}'),
        trailing: coin.marketCapRank != null
            ? Text('Rank: #${coin.marketCapRank}')
            : null,
      ),
    );
  }

  Widget _buildTrendingCard(TrendingCoinModel coin) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: ListTile(
        leading: coin.image.isNotEmpty
            ? Image.network(coin.image, width: 40, height: 40)
            : const Icon(Icons.trending_up),
        title: Text('${coin.name} (${coin.symbol.toUpperCase()})'),
        subtitle: Text('Market Cap Rank: #${coin.marketCapRank}'),
        trailing: Text('Score: ${coin.score}'),
      ),
    );
  }

  Widget _buildGlobalStatsCard() {
    final formatter = NumberFormat.currency(symbol: '\$');
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Global Market Statistics',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Text('Active Cryptocurrencies: ${globalStats!.activeCryptocurrencies}'),
            Text('Markets: ${globalStats!.markets}'),
            Text('Total Market Cap: ${formatter.format(globalStats!.totalMarketCapUsd)}'),
            Text('Total Volume (24h): ${formatter.format(globalStats!.totalVolumeUsd)}'),
            Text('Bitcoin Dominance: ${globalStats!.btcDominance.toStringAsFixed(2)}%'),
            Text('Ethereum Dominance: ${globalStats!.ethDominance.toStringAsFixed(2)}%'),
          ],
        ),
      ),
    );
  }

  Widget _buildWalletHoldingsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Current Holdings',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            if (wallet!.holdings.isEmpty)
              const Text('No holdings yet')
            else
              ...wallet!.holdings.entries.map((entry) {
                final coin = entry.key;
                final holding = entry.value;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4.0),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(coin.toUpperCase()),
                      Text('${holding.qty} @ \$${holding.buyPrice.toStringAsFixed(4)}'),
                    ],
                  ),
                );
              }).toList(),
          ],
        ),
      ),
    );
  }

  Widget _buildPredictionResultCard() {
    final isUp = lastPrediction!.prediction == 'up';
    final color = isUp ? Colors.green : Colors.red;
    final icon = isUp ? Icons.trending_up : Icons.trending_down;
    
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Icon(icon, size: 48, color: color),
            const SizedBox(height: 10),
            Text(
              'Prediction: ${lastPrediction!.prediction.toUpperCase()}',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              'Confidence: ${(lastPrediction!.confidence * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 10),
            Text(
              'Suggested Action: ${isUp ? "BUY" : "SELL"}',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
          ],
        ),
      ),
    );
  }
}