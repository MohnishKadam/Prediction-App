class WalletModel {
  final double balance;
  final Map<String, Holding> holdings;

  WalletModel({required this.balance, required this.holdings});

  factory WalletModel.fromJson(Map<String, dynamic> json) {
    final holdingsMap = <String, Holding>{};
    if (json['holdings'] != null) {
      json['holdings'].forEach((key, value) {
        holdingsMap[key] = Holding.fromJson(value);
      });
    }
    return WalletModel(
      balance: (json['balance'] as num).toDouble(),
      holdings: holdingsMap,
    );
  }
}

class Holding {
  final double qty;
  final double buyPrice;

  Holding({required this.qty, required this.buyPrice});

  factory Holding.fromJson(Map<String, dynamic> json) {
    return Holding(
      qty: (json['qty'] as num).toDouble(),
      buyPrice: (json['buyPrice'] as num).toDouble(),
    );
  }
} 