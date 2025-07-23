class PredictionModel {
  final String prediction;
  final double confidence;

  PredictionModel({required this.prediction, required this.confidence});

  factory PredictionModel.fromJson(Map<String, dynamic> json) {
    return PredictionModel(
      prediction: json['prediction'] as String,
      confidence: (json['confidence'] as num).toDouble(),
    );
  }
}
