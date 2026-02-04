"""
Example Usage of the Interpreter Bot
Demonstrates all features of the bot with practical examples
"""

from interpreter_bot import (
    create_interpreter_bot, 
    explain_prediction_simple,
    explain_error_simple
)
import numpy as np


def example_1_simple_prediction():
    """Example 1: Simple prediction interpretation"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Prediction Interpretation")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    # Different prediction values
    predictions = [0.15, 0.45, 0.75, 0.95]
    
    for pred in predictions:
        print(f"\n--- Prediction: {pred} ---")
        insight = bot.interpret_prediction(pred, context="customer conversion")
        print(insight.user_friendly_description)
        print(f"Confidence: {insight.confidence}")


def example_2_prediction_with_recommendation():
    """Example 2: Prediction with detailed recommendations"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Prediction with Recommendations")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    prediction = 0.72
    actual = 0.68  # For comparison
    
    insight = bot.interpret_prediction(
        prediction=prediction,
        actual_value=actual,
        context="sales forecast for Q4"
    )
    
    print(f"\n📊 Prediction Analysis:")
    print(f"Predicted: {prediction:.2%}")
    print(f"Actual: {actual:.2%}")
    print(f"\n{insight.user_friendly_description}")
    print(f"\nProbability: {insight.probability_explanation}")
    print(f"\nRecommendations:")
    for i, rec in enumerate(insight.recommendations, 1):
        print(f"  {i}. {rec}")


def example_3_error_analysis():
    """Example 3: Detailed error analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Error Analysis")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    # Different error scenarios
    scenarios = [
        (0.85, 0.86, "fraud detection - high confidence"),
        (0.35, 0.65, "customer satisfaction - moderate error"),
        (0.20, 0.80, "risk assessment - severe error"),
    ]
    
    for predicted, actual, context in scenarios:
        print(f"\n--- Predicted: {predicted}, Actual: {actual} ({context}) ---")
        
        insight = bot.interpret_error(predicted, actual, context)
        print(f"Severity: {insight.severity}")
        print(f"Error: {insight.error_percentage:.2f}%")
        print(f"\n{insight.user_friendly_description}")
        
        print(f"\nPossible Causes:")
        for cause in insight.possible_causes[:3]:  # Show first 3
            print(f"  • {cause}")
        
        print(f"\nImprovement Suggestions:")
        for suggestion in insight.improvement_suggestions[:2]:  # Show first 2
            print(f"  ✓ {suggestion}")


def example_4_performance_summary():
    """Example 4: Full performance report generation"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete Performance Summary")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    # Generate realistic predictions and actuals
    np.random.seed(42)
    n_samples = 100
    
    # Predictions closer to actuals = better model
    noise = np.random.normal(0, 0.08, n_samples)  # 8% noise
    actuals = np.random.rand(n_samples)
    predictions = actuals + noise
    predictions = np.clip(predictions, 0, 1)  # Keep in valid range
    
    print(f"\nAnalyzing {n_samples} predictions...")
    summary = bot.generate_performance_summary(predictions, actuals)
    
    print(f"\n📊 Overall Quality: {summary['overall_quality']}")
    print(f"📈 Performance Grade: {summary['performance_grade']}")
    
    print(f"\n📉 Key Metrics:")
    for metric, value in summary['key_metrics'].items():
        print(f"   {metric}: {value}")
    
    print(f"\n💬 Metric Explanations:")
    for key, explanation in summary['interpretation'].items():
        print(f"   • {explanation}")
    
    insights = summary['detailed_insights']
    print(f"\n✨ Best Prediction:")
    print(f"   {insights['best_prediction']['description']}")
    
    print(f"\n💥 Worst Prediction:")
    print(f"   {insights['worst_prediction']['description']}")
    
    print(f"\n🔍 Bias Analysis:")
    print(f"   {insights['bias_analysis']}")


def example_5_trend_analysis():
    """Example 5: Trend analysis from prediction history"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Prediction Trend Analysis")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    # Simulate multiple predictions over time
    prediction_sequence = [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.65, 0.7]
    
    print("\nMaking sequential predictions to track trend...\n")
    
    for i, pred in enumerate(prediction_sequence):
        insight = bot.interpret_prediction(pred, context="performance improvement")
        
        if i > 0:  # Show trend after first prediction
            print(f"Prediction {i+1}: {pred}")
            print(f"  {insight.user_friendly_description}")
            print(f"  {insight.trend_analysis}\n")
        else:
            print(f"Prediction {i+1}: {pred} (baseline)")


def example_6_quick_functions():
    """Example 6: Using quick utility functions"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Quick Utility Functions")
    print("="*70)
    
    # Quick prediction explanation
    print("\n--- Quick Prediction Explanation ---")
    text = explain_prediction_simple(0.85)
    print(text)
    
    # Quick error explanation
    print("\n--- Quick Error Explanation ---")
    text = explain_error_simple(0.70, 0.85)
    print(text)


def example_7_domain_specific():
    """Example 7: Domain-specific contexts"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Domain-Specific Contexts")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    domains = [
        (0.92, "medical diagnosis"),
        (0.45, "stock price movement"),
        (0.78, "equipment failure prediction"),
        (0.15, "spam email detection"),
    ]
    
    for pred, domain in domains:
        insight = bot.interpret_prediction(pred, context=domain)
        print(f"\n[{domain.upper()}]")
        print(f"Prediction: {pred:.0%}")
        print(f"Explanation: {insight.user_friendly_description}")


def example_8_batch_error_analysis():
    """Example 8: Analyzing multiple errors"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Batch Error Analysis")
    print("="*70)
    
    bot = create_interpreter_bot()
    
    # Multiple test cases
    test_cases = [
        (0.9, 0.88, "high confidence - small error"),
        (0.5, 0.7, "uncertain zone - moderate error"),
        (0.2, 0.9, "critical misprediction"),
    ]
    
    for pred, actual, description in test_cases:
        insight = bot.interpret_error(pred, actual, description)
        print(f"\n[{description.upper()}]")
        print(f"Predicted: {pred:.0%} vs Actual: {actual:.0%}")
        print(f"Severity: {insight.severity}")
        print(f"Error: {insight.error_percentage:.1f}%")


def example_9_model_comparison():
    """Example 9: Comparing two models' performance"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Comparing Model Performances")
    print("="*70)
    
    np.random.seed(42)
    n_samples = 50
    actuals = np.random.rand(n_samples)
    
    # Model 1: Better predictions (less noise)
    model1_preds = actuals + np.random.normal(0, 0.03, n_samples)
    model1_preds = np.clip(model1_preds, 0, 1)
    
    # Model 2: Worse predictions (more noise)
    model2_preds = actuals + np.random.normal(0, 0.15, n_samples)
    model2_preds = np.clip(model2_preds, 0, 1)
    
    bot1 = create_interpreter_bot()
    bot2 = create_interpreter_bot()
    
    summary1 = bot1.generate_performance_summary(model1_preds, actuals)
    summary2 = bot2.generate_performance_summary(model2_preds, actuals)
    
    print("\n--- MODEL 1 (Better) ---")
    print(f"Grade: {summary1['performance_grade']}")
    print(f"Quality: {summary1['overall_quality']}")
    print(f"MAE: {summary1['key_metrics']['Mean Absolute Error (MAE)']}")
    
    print("\n--- MODEL 2 (Worse) ---")
    print(f"Grade: {summary2['performance_grade']}")
    print(f"Quality: {summary2['overall_quality']}")
    print(f"MAE: {summary2['key_metrics']['Mean Absolute Error (MAE)']}")
    
    print("\n✅ Model 1 has better performance!")


def example_10_complete_workflow():
    """Example 10: Complete workflow from prediction to analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Complete Workflow")
    print("="*70)
    
    print("\n🔄 Complete ML Pipeline with Interpretation:")
    print("=" * 70)
    
    bot = create_interpreter_bot()
    
    # Step 1: Generate test data
    print("\n1️⃣  STEP 1: Generate Test Data")
    print("-" * 70)
    np.random.seed(42)
    test_X = np.random.randn(10, 2)
    test_y = (test_X[:, 0] + test_X[:, 1] > 0).astype(float)
    print(f"   ✓ Created {len(test_X)} test samples")
    
    # Step 2: Simulate predictions
    print("\n2️⃣  STEP 2: Make Predictions")
    print("-" * 70)
    predictions = np.clip(test_y + np.random.normal(0, 0.1, len(test_y)), 0, 1)
    for i, (pred, actual) in enumerate(zip(predictions[:3], test_y[:3])):
        print(f"   Sample {i+1}: Predicted {pred:.2f}, Actual {actual:.0f}")
    print(f"   ... and {len(predictions) - 3} more")
    
    # Step 3: Interpret each prediction
    print("\n3️⃣  STEP 3: Interpret Predictions")
    print("-" * 70)
    for i in range(min(2, len(predictions))):
        insight = bot.interpret_prediction(predictions[i], context="test data")
        print(f"   Sample {i+1}: {insight.user_friendly_description}")
    
    # Step 4: Analyze errors
    print("\n4️⃣  STEP 4: Analyze Errors")
    print("-" * 70)
    errors = []
    for i, (pred, actual) in enumerate(zip(predictions[:2], test_y[:2])):
        insight = bot.interpret_error(pred, actual)
        errors.append(insight.error_value)
        print(f"   Sample {i+1}: Error={insight.error_percentage:.1f}%, Severity={insight.severity}")
    
    # Step 5: Generate overall report
    print("\n5️⃣  STEP 5: Generate Performance Report")
    print("-" * 70)
    summary = bot.generate_performance_summary(predictions, test_y)
    print(f"   ✓ Overall Quality: {summary['overall_quality']}")
    print(f"   ✓ Performance Grade: {summary['performance_grade']}")
    print(f"   ✓ Accuracy: {summary['key_metrics']['Accuracy']}")
    
    # Step 6: Recommendations
    print("\n6️⃣  STEP 6: Next Steps")
    print("-" * 70)
    if "Poor" in summary['performance_grade'] or "Critical" in summary['overall_quality']:
        print("   ⚠️  Model needs improvement:")
        print("      • Increase training data")
        print("      • Fine-tune hyperparameters")
        print("      • Add more hidden neurons")
    else:
        print("   ✅ Model is performing well!")
        print("      • Monitor on new data")
        print("      • Consider deployment")


def main():
    """Run all examples"""
    print("\n" + "🤖 " * 20)
    print("INTERPRETER BOT - COMPREHENSIVE EXAMPLES")
    print("🤖 " * 20)
    
    examples = [
        ("Simple Prediction", example_1_simple_prediction),
        ("Prediction with Recommendation", example_2_prediction_with_recommendation),
        ("Error Analysis", example_3_error_analysis),
        ("Performance Summary", example_4_performance_summary),
        ("Trend Analysis", example_5_trend_analysis),
        ("Quick Functions", example_6_quick_functions),
        ("Domain-Specific", example_7_domain_specific),
        ("Batch Error Analysis", example_8_batch_error_analysis),
        ("Model Comparison", example_9_model_comparison),
        ("Complete Workflow", example_10_complete_workflow),
    ]
    
    # Run all examples
    for title, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {title}: {e}")
    
    print("\n" + "🤖 " * 20)
    print("ALL EXAMPLES COMPLETED!")
    print("🤖 " * 20 + "\n")


if __name__ == "__main__":
    main()
