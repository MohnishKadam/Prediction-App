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
  static const String _baseUrl = "https://api.coingecko.com/api/v3";

  // Get wallet
  static Future<WalletModel> getWallet() async {
    final response = await http.get(Uri.parse('$_baseUrl/wallet'));
    if (response.statusCode == 200) {
      return WalletModel.fromJson(json.decode(response.body));
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Fetch and predict (quick prediction)
  static Future<PredictionModel> fetchAndPredict(String coinId) async {
    final response = await http.post(
      Uri.parse('http://192.168.1.7:5000/fetch-and-predict'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'id': coinId}),
    );
    if (response.statusCode == 200) {
      return PredictionModel.fromJson(json.decode(response.body));
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Predict (historical, more accurate)
  static Future<PredictionModel> predict(String coinId) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'id': coinId}),
    );
    if (response.statusCode == 200) {
      return PredictionModel.fromJson(json.decode(response.body));
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Buy
  static Future<WalletModel> buy(String coinId, double qty) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/buy'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'coin': coinId, 'qty': qty}),
    );
    if (response.statusCode == 200) {
      return WalletModel.fromJson(json.decode(response.body));
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Sell
  static Future<WalletModel> sell(String coinId, double qty) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/sell'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'coin': coinId, 'qty': qty}),
    );
    if (response.statusCode == 200) {
      return WalletModel.fromJson(json.decode(response.body));
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Fetch market data
  static Future<List<MarketCoinModel>> fetchMarketData({
    String currency = 'usd',
    int perPage = 10,
  }) async {
    final response = await http.get(Uri.parse(
      '$_baseUrl/coins/markets?vs_currency=$currency&order=market_cap_desc&per_page=$perPage&page=1&sparkline=false',
    ));
    if (response.statusCode == 200) {
      final List<dynamic> data = json.decode(response.body);
      return data.map((e) => MarketCoinModel.fromJson(e)).toList();
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Search coins
  static Future<List<SearchCoinModel>> searchCoins(String query) async {
    final response = await http.get(Uri.parse('$_baseUrl/search?query=$query'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final List<dynamic> coins = data['coins'] ?? [];
      return coins.map((e) => SearchCoinModel.fromJson(e)).toList();
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Fetch trending coins
  static Future<List<TrendingCoinModel>> fetchTrendingCoins() async {
    final response = await http.get(Uri.parse('$_baseUrl/search/trending'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      final List<dynamic> coins = data['coins'] ?? [];
      return coins.map((e) => TrendingCoinModel.fromJson(e)).toList();
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }

  // Fetch global stats
  static Future<GlobalStatsModel> fetchGlobalStats() async {
    final response = await http.get(Uri.parse('$_baseUrl/global'));
    if (response.statusCode == 200) {
      final data = json.decode(response.body);
      return GlobalStatsModel.fromJson(data);
    } else {
      throw ErrorModel.fromJson(json.decode(response.body));
    }
  }
}
