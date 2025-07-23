class GlobalStatsModel {
  final int activeCryptocurrencies;
  final int markets;
  final double totalMarketCapUsd;
  final double totalVolumeUsd;
  final double btcDominance;
  final double ethDominance;

  GlobalStatsModel({
    required this.activeCryptocurrencies,
    required this.markets,
    required this.totalMarketCapUsd,
    required this.totalVolumeUsd,
    required this.btcDominance,
    required this.ethDominance,
  });

  factory GlobalStatsModel.fromJson(Map<String, dynamic> json) {
    final data = json['data'] ?? json;
    return GlobalStatsModel(
      activeCryptocurrencies: data['active_cryptocurrencies'] as int? ?? 0,
      markets: data['markets'] as int? ?? 0,
      totalMarketCapUsd:
          (data['total_market_cap']?['usd'] as num?)?.toDouble() ?? 0.0,
      totalVolumeUsd: (data['total_volume']?['usd'] as num?)?.toDouble() ?? 0.0,
      btcDominance:
          (data['market_cap_percentage']?['btc'] as num?)?.toDouble() ?? 0.0,
      ethDominance:
          (data['market_cap_percentage']?['eth'] as num?)?.toDouble() ?? 0.0,
    );
  }
}
