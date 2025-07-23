class TrendingCoinModel {
  final String id;
  final String name;
  final String symbol;
  final String image;
  final int marketCapRank;
  final int score;

  TrendingCoinModel({
    required this.id,
    required this.name,
    required this.symbol,
    required this.image,
    required this.marketCapRank,
    required this.score,
  });

  factory TrendingCoinModel.fromJson(Map<String, dynamic> json) {
    final item = json['item'] ?? json;
    return TrendingCoinModel(
      id: item['id'] as String,
      name: item['name'] as String,
      symbol: item['symbol'] as String,
      image: item['large'] ?? item['thumb'] ?? '',
      marketCapRank: item['market_cap_rank'] as int? ?? 0,
      score: item['score'] as int? ?? 0,
    );
  }
}
