class SearchCoinModel {
  final String id;
  final String name;
  final String symbol;
  final String image;
  final int? marketCapRank;

  SearchCoinModel({
    required this.id,
    required this.name,
    required this.symbol,
    required this.image,
    this.marketCapRank,
  });

  factory SearchCoinModel.fromJson(Map<String, dynamic> json) {
    return SearchCoinModel(
      id: json['id'] as String,
      name: json['name'] as String,
      symbol: json['symbol'] as String,
      image: json['large'] ?? json['thumb'] ?? '',
      marketCapRank: json['market_cap_rank'] as int?,
    );
  }
}
