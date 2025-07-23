import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/wallet_model.dart';
import '../models/prediction_model.dart';
import '../models/error_model.dart';
import '../models/market_coin_model.dart';
import '../models/search_coin_model.dart';
import '../models/trending_coin_model.dart';
import '../models/global_stats_model.dart';

class ApiService {
  // Update this to match your Flask backend URL
  static const String _flaskBaseUrl = "http://localhost:5000";
  static const String _coinGeckoBaseUrl = "https://api.coingecko.com/api/v3";
  static const String _apiKey = 'CG-C9MWDmLXa1Dd4yxxTvAWtMdz';

  // Flask backend endpoints
  static Future<WalletModel> getWallet() async {
    try {
      final response = await http.get(
        Uri.parse('$_flaskBaseUrl/wallet'),
        headers: {'Content-Type': 'application/json'},
      );
      
      if (response.statusCode == 200) {
        return WalletModel.fromJson(json.decode(response.body));
      } else {
        throw Exception('Failed to get wallet: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  // Quick prediction using current market data
  static Future<PredictionModel> fetchAndPredict(String coinId) async {
    try {
      final response = await http.post(
        Uri.parse('$_flaskBaseUrl/fetch-and-predict'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'id': coinId}),
      );
      
      if (response.statusCode == 200) {
        return PredictionModel.fromJson(json.decode(response.body));
      } else {
        final error = json.decode(response.body);
        throw Exception(error['error'] ?? 'Prediction failed');
      }
    } catch (e) {
      throw Exception('Prediction error: $e');
    }
  }

  // Historical prediction (more accurate)
  static Future<PredictionModel> predict(String coinId) async {
    try {
      final response = await http.post(
        Uri.parse('$_flaskBaseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'id': coinId}),
      );
      
      if (response.statusCode == 200) {
        return PredictionModel.fromJson(json.decode(response.body));
      } else {
        final error = json.decode(response.body);
        throw Exception(error['error'] ?? 'Prediction failed');
      }
    } catch (e) {
      throw Exception('Prediction error: $e');
    }
  }

  // Buy cryptocurrency
  static Future<WalletModel> buy(String coinId, double qty) async {
    try {
      final response = await http.post(
        Uri.parse('$_flaskBaseUrl/buy'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'coin': coinId, 'qty': qty}),
      );
      
      if (response.statusCode == 200) {
        return WalletModel.fromJson(json.decode(response.body));
      } else {
        final error = json.decode(response.body);
        throw Exception(error['error'] ?? 'Buy failed');
      }
    } catch (e) {
      throw Exception('Buy error: $e');
    }
  }

  // Sell cryptocurrency
  static Future<WalletModel> sell(String coinId, double qty) async {
    try {
      final response = await http.post(
        Uri.parse('$_flaskBaseUrl/sell'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'coin': coinId, 'qty': qty}),
      );
      
      if (response.statusCode == 200) {
        return WalletModel.fromJson(json.decode(response.body));
      } else {
        final error = json.decode(response.body);
        throw Exception(error['error'] ?? 'Sell failed');
      }
    } catch (e) {
      throw Exception('Sell error: $e');
    }
  }

  // CoinGecko API endpoints (for market data)
  static Future<List<MarketCoinModel>> fetchMarketData({
    String currency = 'usd',
    int perPage = 10,
  }) async {
    try {
      final response = await http.get(
        Uri.parse(
          '$_coinGeckoBaseUrl/coins/markets?vs_currency=$currency&order=market_cap_desc&per_page=$perPage&page=1&sparkline=false',
        ),
        headers: {'x-cg-demo-api-key': _apiKey},
      );
      
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        return data.map((e) => MarketCoinModel.fromJson(e)).toList();
      } else {
        throw Exception('Failed to fetch market data: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Market data error: $e');
    }
  }

  // Search coins
  static Future<List<SearchCoinModel>> searchCoins(String query) async {
    try {
      final response = await http.get(
        Uri.parse('$_coinGeckoBaseUrl/search?query=$query'),
        headers: {'x-cg-demo-api-key': _apiKey},
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final List<dynamic> coins = data['coins'] ?? [];
        return coins.map((e) => SearchCoinModel.fromJson(e)).toList();
      } else {
        throw Exception('Failed to search coins: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Search error: $e');
    }
  }

  // Fetch trending coins
  static Future<List<TrendingCoinModel>> fetchTrendingCoins() async {
    try {
      final response = await http.get(
        Uri.parse('$_coinGeckoBaseUrl/search/trending'),
        headers: {'x-cg-demo-api-key': _apiKey},
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final List<dynamic> coins = data['coins'] ?? [];
        return coins.map((e) => TrendingCoinModel.fromJson(e)).toList();
      } else {
        throw Exception('Failed to fetch trending coins: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Trending coins error: $e');
    }
  }

  // Fetch global stats
  static Future<GlobalStatsModel> fetchGlobalStats() async {
    try {
      final response = await http.get(
        Uri.parse('$_coinGeckoBaseUrl/global'),
        headers: {'x-cg-demo-api-key': _apiKey},
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return GlobalStatsModel.fromJson(data);
      } else {
        throw Exception('Failed to fetch global stats: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Global stats error: $e');
    }
  }

  // Get current price for a specific coin
  static Future<double> getCurrentPrice(String coinId) async {
    try {
      final response = await http.get(
        Uri.parse('$_coinGeckoBaseUrl/simple/price?ids=$coinId&vs_currencies=usd'),
        headers: {'x-cg-demo-api-key': _apiKey},
      );
      
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return (data[coinId]?['usd'] as num?)?.toDouble() ?? 0.0;
      } else {
        throw Exception('Failed to get current price: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Price fetch error: $e');
    }
  }
}