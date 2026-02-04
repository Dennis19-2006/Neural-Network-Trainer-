"""
Unit Tests for Interpreter Bot
Tests all functionality and ensures correct behavior
"""

import unittest
import numpy as np
from interpreter_bot import (
    InterpreterBot,
    create_interpreter_bot,
    explain_prediction_simple,
    explain_error_simple,
    PredictionInsight,
    ErrorInsight,
    ConfidenceLevel
)


class TestInterpreterBotBasics(unittest.TestCase):
    """Test basic functionality of interpreter bot"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.bot = create_interpreter_bot()
    
    def test_bot_creation(self):
        """Test bot can be created"""
        self.assertIsNotNone(self.bot)
        self.assertIsInstance(self.bot, InterpreterBot)
    
    def test_prediction_interpretation_returns_insight(self):
        """Test prediction interpretation returns proper object"""
        insight = self.bot.interpret_prediction(0.7)
        self.assertIsInstance(insight, PredictionInsight)
        self.assertEqual(insight.prediction, 0.7)
    
    def test_error_interpretation_returns_insight(self):
        """Test error interpretation returns proper object"""
        insight = self.bot.interpret_error(0.7, 0.9)
        self.assertIsInstance(insight, ErrorInsight)
        self.assertAlmostEqual(insight.error_value, -0.2)


class TestPredictionInterpretation(unittest.TestCase):
    """Test prediction interpretation features"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
    
    def test_confidence_level_very_low(self):
        """Test confidence categorization for very low predictions"""
        insight = self.bot.interpret_prediction(0.1)
        self.assertEqual(insight.confidence, "Very Low")
    
    def test_confidence_level_very_high(self):
        """Test confidence categorization for very high predictions"""
        insight = self.bot.interpret_prediction(0.9)
        self.assertEqual(insight.confidence, "Very High")
    
    def test_prediction_description_contains_percentage(self):
        """Test prediction description contains percentage"""
        insight = self.bot.interpret_prediction(0.75)
        self.assertIn("75", insight.user_friendly_description)
    
    def test_prediction_description_contains_context(self):
        """Test prediction description includes provided context"""
        context = "fraud detection"
        insight = self.bot.interpret_prediction(0.8, context=context)
        self.assertIn(context, insight.user_friendly_description)
    
    def test_probability_explanation_exists(self):
        """Test probability explanation is generated"""
        insight = self.bot.interpret_prediction(0.5)
        self.assertIsNotNone(insight.probability_explanation)
        self.assertTrue(len(insight.probability_explanation) > 0)
    
    def test_recommendations_generated(self):
        """Test recommendations are generated"""
        insight = self.bot.interpret_prediction(0.6)
        self.assertIsInstance(insight.recommendations, list)
        self.assertGreater(len(insight.recommendations), 0)
    
    def test_prediction_clipping(self):
        """Test predictions outside [0,1] are clipped"""
        insight = self.bot.interpret_prediction(1.5)
        self.assertLessEqual(insight.prediction, 1.0)
        
        insight = self.bot.interpret_prediction(-0.5)
        self.assertGreaterEqual(insight.prediction, 0.0)
    
    def test_prediction_history_tracking(self):
        """Test bot tracks prediction history"""
        initial_len = len(self.bot.prediction_history)
        self.bot.interpret_prediction(0.5)
        self.assertEqual(len(self.bot.prediction_history), initial_len + 1)


class TestErrorInterpretation(unittest.TestCase):
    """Test error interpretation features"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
    
    def test_error_severity_excellent(self):
        """Test error severity categorization for excellent predictions"""
        insight = self.bot.interpret_error(0.95, 0.96)
        self.assertIn("Excellent", insight.severity)
    
    def test_error_severity_critical(self):
        """Test error severity categorization for critical errors"""
        insight = self.bot.interpret_error(0.1, 0.9)
        self.assertIn("Critical", insight.severity)
    
    def test_error_calculation(self):
        """Test error is calculated correctly"""
        predicted = 0.6
        actual = 0.8
        insight = self.bot.interpret_error(predicted, actual)
        self.assertAlmostEqual(insight.error_value, actual - predicted)
    
    def test_error_percentage_calculation(self):
        """Test error percentage is calculated"""
        insight = self.bot.interpret_error(0.5, 1.0)
        self.assertGreater(insight.error_percentage, 0)
    
    def test_error_causes_identified(self):
        """Test error causes are identified"""
        insight = self.bot.interpret_error(0.1, 0.9)
        self.assertIsInstance(insight.possible_causes, list)
        self.assertGreater(len(insight.possible_causes), 0)
    
    def test_improvement_suggestions_provided(self):
        """Test improvement suggestions are provided"""
        insight = self.bot.interpret_error(0.2, 0.8)
        self.assertIsInstance(insight.improvement_suggestions, list)
        self.assertGreater(len(insight.improvement_suggestions), 0)
    
    def test_error_history_tracking(self):
        """Test bot tracks error history"""
        initial_len = len(self.bot.error_history)
        self.bot.interpret_error(0.5, 0.7)
        self.assertEqual(len(self.bot.error_history), initial_len + 1)
    
    def test_error_description_indicates_direction(self):
        """Test error description indicates over/underestimation"""
        insight_under = self.bot.interpret_error(0.5, 0.8)  # Underestimated
        self.assertIn("underestimated", insight_under.user_friendly_description.lower())
        
        insight_over = self.bot.interpret_error(0.8, 0.5)  # Overestimated
        self.assertIn("overestimated", insight_over.user_friendly_description.lower())


class TestPerformanceSummary(unittest.TestCase):
    """Test performance summary generation"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
        np.random.seed(42)
    
    def test_summary_returns_dict(self):
        """Test summary returns dictionary"""
        predictions = [0.5, 0.7, 0.3, 0.9]
        actuals = [0.5, 0.7, 0.3, 0.9]
        summary = self.bot.generate_performance_summary(predictions, actuals)
        self.assertIsInstance(summary, dict)
    
    def test_summary_contains_key_fields(self):
        """Test summary contains expected fields"""
        predictions = np.random.rand(10)
        actuals = np.random.rand(10)
        summary = self.bot.generate_performance_summary(predictions, actuals)
        
        self.assertIn('overall_quality', summary)
        self.assertIn('key_metrics', summary)
        self.assertIn('interpretation', summary)
        self.assertIn('performance_grade', summary)
        self.assertIn('detailed_insights', summary)
    
    def test_perfect_predictions_get_high_grade(self):
        """Test perfect predictions receive high grade"""
        predictions = [0.1, 0.5, 0.9, 0.3, 0.7]
        actuals = [0.1, 0.5, 0.9, 0.3, 0.7]  # Perfect match
        summary = self.bot.generate_performance_summary(predictions, actuals)
        self.assertIn('A', summary['performance_grade'])
    
    def test_poor_predictions_get_low_grade(self):
        """Test poor predictions receive low grade"""
        predictions = [0.1, 0.2, 0.3, 0.4, 0.5]
        actuals = [0.9, 0.8, 0.7, 0.6, 0.5]  # All wrong
        summary = self.bot.generate_performance_summary(predictions, actuals)
        # Should be lower than A
        self.assertNotIn('A+', summary['performance_grade'])
    
    def test_key_metrics_calculated(self):
        """Test key metrics are calculated"""
        predictions = np.array([0.5, 0.7, 0.3])
        actuals = np.array([0.6, 0.7, 0.4])
        summary = self.bot.generate_performance_summary(predictions, actuals)
        
        metrics = summary['key_metrics']
        self.assertIn('Mean Absolute Error (MAE)', metrics)
        self.assertIn('Root Mean Squared Error (RMSE)', metrics)
        self.assertIn('Accuracy', metrics)
    
    def test_best_worst_predictions_identified(self):
        """Test best and worst predictions are identified"""
        predictions = [0.1, 0.5, 0.9, 0.2, 0.8]
        actuals = [0.1, 0.5, 0.9, 0.2, 0.8]  # Perfect
        summary = self.bot.generate_performance_summary(predictions, actuals)
        
        insights = summary['detailed_insights']
        self.assertIn('best_prediction', insights)
        self.assertIn('worst_prediction', insights)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_explain_prediction_simple(self):
        """Test simple prediction explanation function"""
        text = explain_prediction_simple(0.75)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
    
    def test_explain_error_simple(self):
        """Test simple error explanation function"""
        text = explain_error_simple(0.6, 0.9)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
    
    def test_prediction_at_boundaries(self):
        """Test predictions at 0 and 1 boundaries"""
        insight_0 = self.bot.interpret_prediction(0.0)
        insight_1 = self.bot.interpret_prediction(1.0)
        
        self.assertEqual(insight_0.prediction, 0.0)
        self.assertEqual(insight_1.prediction, 1.0)
    
    def test_zero_error(self):
        """Test handling of zero error"""
        insight = self.bot.interpret_error(0.5, 0.5)
        self.assertAlmostEqual(insight.error_value, 0.0)
    
    def test_empty_history_analysis(self):
        """Test behavior with empty history"""
        new_bot = InterpreterBot()
        insight = new_bot.interpret_prediction(0.5)
        # Should handle gracefully
        self.assertIsNotNone(insight.trend_analysis)
    
    def test_single_sample_performance(self):
        """Test performance summary with single sample"""
        summary = self.bot.generate_performance_summary([0.5], [0.5])
        self.assertIsNotNone(summary)
        self.assertGreater(len(summary), 0)


class TestFormatting(unittest.TestCase):
    """Test output formatting"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
    
    def test_format_prediction_insight(self):
        """Test formatting of prediction insight"""
        insight = self.bot.interpret_prediction(0.7)
        formatted = self.bot.format_insights_for_display(insight)
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)
    
    def test_format_error_insight(self):
        """Test formatting of error insight"""
        insight = self.bot.interpret_error(0.6, 0.9)
        formatted = self.bot.format_insights_for_display(insight)
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)


class TestBiasDetection(unittest.TestCase):
    """Test bias detection in models"""
    
    def setUp(self):
        self.bot = create_interpreter_bot()
    
    def test_systematic_overestimation_detection(self):
        """Test detection of systematic overestimation"""
        predictions = [0.9, 0.9, 0.9, 0.9, 0.9]
        actuals = [0.3, 0.3, 0.3, 0.3, 0.3]
        summary = self.bot.generate_performance_summary(predictions, actuals)
        
        bias = summary['detailed_insights']['bias_analysis']
        self.assertIn('overestimate', bias.lower())
    
    def test_systematic_underestimation_detection(self):
        """Test detection of systematic underestimation"""
        predictions = [0.1, 0.1, 0.1, 0.1, 0.1]
        actuals = [0.9, 0.9, 0.9, 0.9, 0.9]
        summary = self.bot.generate_performance_summary(predictions, actuals)
        
        bias = summary['detailed_insights']['bias_analysis']
        self.assertIn('underestimate', bias.lower())


def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RUNNING INTERPRETER BOT TESTS")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestInterpreterBotBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionInterpretation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorInterpretation))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestFormatting))
    suite.addTests(loader.loadTestsFromTestCase(TestBiasDetection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
