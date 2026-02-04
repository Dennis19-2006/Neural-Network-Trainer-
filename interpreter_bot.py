"""
Interpreter/Descriptor Bot for Neural Network Predictions and Errors
Explains predictions and errors in user-friendly language
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    VERY_LOW = (0, 0.3, "Very Low")
    LOW = (0.3, 0.5, "Low")
    MEDIUM = (0.5, 0.7, "Medium")
    HIGH = (0.7, 0.85, "High")
    VERY_HIGH = (0.85, 1.0, "Very High")


@dataclass
class PredictionInsight:
    """Container for prediction insights"""
    prediction: float
    confidence: str
    user_friendly_description: str
    probability_explanation: str
    trend_analysis: str
    recommendations: List[str]


@dataclass
class ErrorInsight:
    """Container for error insights"""
    error_value: float
    error_percentage: float
    severity: str
    user_friendly_description: str
    possible_causes: List[str]
    improvement_suggestions: List[str]
    comparison_context: str


class InterpreterBot:
    """
    An intelligent bot that interprets neural network predictions and errors
    and explains them in user-friendly language
    """

    def __init__(self):
        """Initialize the interpreter bot"""
        self.error_history = []
        self.prediction_history = []
        self.model_performance_baseline = None

    # ============================================================================
    # PREDICTION INTERPRETATION
    # ============================================================================

    def interpret_prediction(
        self,
        prediction: float,
        actual_value: float = None,
        input_features: List[float] = None,
        feature_names: List[str] = None,
        context: str = "unknown"
    ) -> PredictionInsight:
        """
        Interpret a neural network prediction and provide user-friendly explanation

        Args:
            prediction: The model's prediction value (0-1 for sigmoid output)
            actual_value: Optional actual value for comparison
            input_features: Optional input features that led to this prediction
            feature_names: Optional names of the input features
            context: Context description for the prediction

        Returns:
            PredictionInsight object with detailed explanation
        """
        
        # Clip prediction to valid range
        pred_clipped = np.clip(prediction, 0, 1)
        
        # Determine confidence level
        confidence = self._categorize_confidence(pred_clipped)
        
        # Generate user-friendly description
        description = self._generate_prediction_description(pred_clipped, context)
        
        # Generate probability explanation
        probability_text = self._explain_probability(pred_clipped)
        
        # Analyze trend if we have history
        trend_analysis = self._analyze_prediction_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(pred_clipped, actual_value)
        
        # Store in history
        self.prediction_history.append({
            "value": pred_clipped,
            "actual": actual_value,
            "context": context
        })
        
        return PredictionInsight(
            prediction=pred_clipped,
            confidence=confidence,
            user_friendly_description=description,
            probability_explanation=probability_text,
            trend_analysis=trend_analysis,
            recommendations=recommendations
        )

    def _categorize_confidence(self, prediction: float) -> str:
        """Determine confidence level from prediction value"""
        distance_from_boundary = min(abs(prediction - 0.5), 0.5)
        confidence_score = distance_from_boundary * 2  # 0 to 1
        
        for level in ConfidenceLevel:
            if level.value[0] <= confidence_score < level.value[1]:
                return level.value[2]
        return "Very High"

    def _generate_prediction_description(self, prediction: float, context: str) -> str:
        """Generate a simple description of what the prediction means"""
        
        percentage = prediction * 100
        
        if prediction < 0.2:
            return f"🔴 **Very Unlikely** ({percentage:.1f}%): The model strongly predicts this will NOT happen ({context})."
        elif prediction < 0.4:
            return f"🟡 **Unlikely** ({percentage:.1f}%): The model suggests this is probably not going to happen ({context})."
        elif prediction < 0.6:
            return f"🟠 **Uncertain** ({percentage:.1f}%): The model is unsure - it could go either way ({context})."
        elif prediction < 0.8:
            return f"🟢 **Likely** ({percentage:.1f}%): The model predicts this is probably going to happen ({context})."
        else:
            return f"🟢 **Very Likely** ({percentage:.1f}%): The model strongly predicts this will happen ({context})."

    def _explain_probability(self, prediction: float) -> str:
        """Explain the probability in simple terms"""
        percentage = prediction * 100
        
        if percentage < 1:
            return f"Almost certainly won't happen - only {percentage:.2f}% chance."
        elif percentage < 10:
            return f"Very unlikely - only a {percentage:.1f}% chance of occurring."
        elif percentage < 30:
            return f"Unlikely but possible - about {percentage:.1f}% chance."
        elif percentage < 50:
            return f"Less likely than not - {percentage:.1f}% probability."
        elif percentage < 70:
            return f"More likely than not - {percentage:.1f}% probability."
        elif percentage < 90:
            return f"Very likely to happen - {percentage:.1f}% probability."
        else:
            return f"Almost certainly will happen - {percentage:.1f}% probability."

    def _analyze_prediction_trend(self) -> str:
        """Analyze trend in recent predictions"""
        if len(self.prediction_history) < 2:
            return "Not enough prediction history to analyze trends yet."
        
        recent_preds = [p["value"] for p in self.prediction_history[-5:]]
        avg_recent = np.mean(recent_preds)
        avg_before = np.mean([p["value"] for p in self.prediction_history[:-5]]) if len(self.prediction_history) > 5 else avg_recent
        
        change = avg_recent - avg_before
        
        if abs(change) < 0.02:
            return "📊 Trend: Stable predictions - model is consistent in its assessments."
        elif change > 0:
            return f"📈 Trend: Increasing predictions - model becoming more confident in positive outcomes (change: +{change:.2%})."
        else:
            return f"📉 Trend: Decreasing predictions - model becoming more conservative (change: {change:.2%})."

    def _generate_recommendations(self, prediction: float, actual_value: float = None) -> List[str]:
        """Generate actionable recommendations based on prediction"""
        recommendations = []
        
        if prediction < 0.3:
            recommendations.append("✅ Take preventive action - the risk is low but monitor for changes.")
            recommendations.append("📋 Consider alternative strategies - current approach may not be effective.")
        elif prediction < 0.7:
            recommendations.append("⚠️ Stay alert - outcomes are uncertain, be prepared for both scenarios.")
            recommendations.append("🔍 Gather more information - additional data may improve prediction clarity.")
        else:
            recommendations.append("✅ Proceed with confidence - prediction is favorable.")
            recommendations.append("📊 Monitor actual results - verify that predictions align with reality.")
        
        if actual_value is not None:
            error = abs(prediction - actual_value)
            if error < 0.1:
                recommendations.append("🎯 Excellent prediction accuracy - model is performing well!")
            elif error < 0.3:
                recommendations.append("⚠️ Moderate prediction error - model needs refinement in this scenario.")
            else:
                recommendations.append("🔧 Large prediction error - model may need retraining with more diverse data.")
        
        return recommendations

    # ============================================================================
    # ERROR INTERPRETATION
    # ============================================================================

    def interpret_error(
        self,
        predicted: float,
        actual: float,
        sample_context: str = "unknown"
    ) -> ErrorInsight:
        """
        Interpret a prediction error and explain it in user-friendly terms

        Args:
            predicted: Model's prediction
            actual: Actual value
            sample_context: Context for this error

        Returns:
            ErrorInsight object with detailed explanation
        """
        
        error = actual - predicted
        abs_error = abs(error)
        error_percentage = abs(error / (actual + 1e-10)) * 100  # Avoid division by zero
        
        # Determine severity
        severity = self._categorize_error_severity(abs_error)
        
        # Generate description
        description = self._describe_error(error, abs_error)
        
        # Find possible causes
        causes = self._identify_error_causes(error, predicted, actual)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(severity, abs_error)
        
        # Provide context comparison
        context = self._compare_error_context(abs_error, sample_context)
        
        # Store in history
        self.error_history.append({
            "absolute": abs_error,
            "percentage": error_percentage,
            "severity": severity
        })
        
        return ErrorInsight(
            error_value=error,
            error_percentage=error_percentage,
            severity=severity,
            user_friendly_description=description,
            possible_causes=causes,
            improvement_suggestions=suggestions,
            comparison_context=context
        )

    def _categorize_error_severity(self, error: float) -> str:
        """Categorize error severity"""
        if error < 0.05:
            return "Excellent ✅"
        elif error < 0.1:
            return "Good ✅"
        elif error < 0.2:
            return "Acceptable ⚠️"
        elif error < 0.35:
            return "Poor ❌"
        else:
            return "Critical ❌"

    def _describe_error(self, error: float, abs_error: float) -> str:
        """Describe the error in simple terms"""
        error_pct = abs_error * 100
        
        direction = "overestimated" if error > 0 else "underestimated"
        
        if abs_error < 0.05:
            return f"🎯 Excellent prediction! Model only {direction} by {error_pct:.2f}% - nearly perfect."
        elif abs_error < 0.1:
            return f"✅ Good prediction. Model {direction} by {error_pct:.1f}% - minor discrepancy."
        elif abs_error < 0.2:
            return f"⚠️ Acceptable prediction. Model {direction} by {error_pct:.1f}% - noticeable but tolerable error."
        elif abs_error < 0.35:
            return f"❌ Poor prediction. Model {direction} by {error_pct:.1f}% - significant mismatch."
        else:
            return f"❌ Critical error. Model {direction} by {error_pct:.1f}% - severe prediction failure."

    def _identify_error_causes(self, error: float, predicted: float, actual: float) -> List[str]:
        """Identify possible causes of prediction error"""
        causes = []
        
        if abs(error) > 0.3:
            causes.append("🔴 **Insufficient training data** - model may not have seen similar examples.")
            causes.append("🔴 **Feature mismatch** - input features might not capture important patterns.")
            causes.append("🔴 **Model complexity** - network might be too simple or too complex for this problem.")
        
        if error > 0:  # Overestimation
            causes.append("📊 Model tends to **overestimate** - might be biased toward positive predictions.")
            causes.append("⚙️ **Learning rate might be too high** - weights updated too aggressively.")
        else:  # Underestimation
            causes.append("📊 Model tends to **underestimate** - might be biased toward conservative predictions.")
            causes.append("⚙️ **Network might need more hidden neurons** - insufficient model capacity.")
        
        if predicted < 0.1 or predicted > 0.9:
            causes.append("🟡 **Extreme predictions** - model is very confident, leaving little room for error.")
        
        if 0.4 < predicted < 0.6:
            causes.append("🟡 **Uncertain prediction zone** - model was already unsure about this sample.")
        
        return causes

    def _generate_improvement_suggestions(self, severity: str, error: float) -> List[str]:
        """Generate suggestions to improve model performance"""
        suggestions = []
        
        if "Critical" in severity or "Poor" in severity:
            suggestions.append("🔄 **Retrain the model** with more epochs or higher learning rate.")
            suggestions.append("📈 **Add more training data** - especially examples similar to this error case.")
            suggestions.append("🧠 **Increase hidden neurons** - model might have insufficient capacity.")
            suggestions.append("🔍 **Review input features** - ensure they are meaningful and well-scaled.")
        elif "Acceptable" in severity:
            suggestions.append("⚡ **Fine-tune hyperparameters** - small adjustments might help.")
            suggestions.append("📊 **Analyze feature importance** - focus on most predictive features.")
        else:
            suggestions.append("✅ Keep current configuration - model is performing well.")
            suggestions.append("📈 **Monitor on new data** - ensure performance remains consistent.")
        
        return suggestions

    def _compare_error_context(self, error: float, context: str) -> str:
        """Provide context on how this error compares to typical performance"""
        avg_error = np.mean([e["absolute"] for e in self.error_history]) if self.error_history else error
        
        if len(self.error_history) == 0:
            return f"This is the first error recorded ({context})."
        
        comparison = error / avg_error if avg_error > 0 else 1
        
        if comparison < 0.8:
            return f"✅ Better than average - this error ({error:.4f}) is below the average error of {avg_error:.4f} ({context})."
        elif comparison < 1.2:
            return f"📊 Around average - this error ({error:.4f}) is similar to the average error of {avg_error:.4f} ({context})."
        else:
            return f"❌ Worse than average - this error ({error:.4f}) exceeds the average error of {avg_error:.4f} ({context})."

    # ============================================================================
    # MODEL PERFORMANCE SUMMARY
    # ============================================================================

    def generate_performance_summary(self, predictions: List[float], actuals: List[float]) -> Dict:
        """
        Generate a comprehensive, user-friendly performance summary

        Args:
            predictions: List of model predictions
            actuals: List of actual values

        Returns:
            Dictionary with performance metrics and explanations
        """
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        errors = actuals - predictions
        abs_errors = np.abs(errors)
        
        mae = np.mean(abs_errors)
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy for binary classification (if values close to 0 or 1)
        binary_preds = np.round(predictions)
        accuracy = np.mean(binary_preds == actuals) * 100
        
        # Generate summary
        summary = {
            "overall_quality": self._evaluate_overall_quality(mae, rmse),
            "key_metrics": {
                "Mean Absolute Error (MAE)": f"{mae:.4f}",
                "Root Mean Squared Error (RMSE)": f"{rmse:.4f}",
                "Accuracy": f"{accuracy:.1f}%"
            },
            "interpretation": {
                "mae_explanation": self._explain_metric(mae, "MAE", "On average, predictions are off by"),
                "rmse_explanation": self._explain_metric(rmse, "RMSE", "Typical prediction error magnitude is"),
                "accuracy_explanation": self._explain_accuracy(accuracy)
            },
            "performance_grade": self._assign_performance_grade(accuracy, mae),
            "detailed_insights": self._generate_detailed_insights(predictions, actuals, errors)
        }
        
        return summary

    def _evaluate_overall_quality(self, mae: float, rmse: float) -> str:
        """Evaluate overall model quality"""
        if mae < 0.05 and rmse < 0.1:
            return "🌟 Excellent - Model is performing exceptionally well!"
        elif mae < 0.1 and rmse < 0.2:
            return "✅ Good - Model is performing well with acceptable accuracy."
        elif mae < 0.2 and rmse < 0.35:
            return "⚠️ Fair - Model has moderate accuracy, room for improvement."
        elif mae < 0.35:
            return "❌ Poor - Model needs significant retraining and optimization."
        else:
            return "❌ Critical - Model requires major changes or retraining."

    def _explain_metric(self, value: float, metric_name: str, prefix: str) -> str:
        """Explain what a metric value means"""
        percentage = value * 100
        
        if value < 0.05:
            return f"{prefix} {percentage:.2f}% - excellent precision!"
        elif value < 0.1:
            return f"{prefix} {percentage:.1f}% - good accuracy."
        elif value < 0.2:
            return f"{prefix} {percentage:.1f}% - acceptable but could be better."
        elif value < 0.35:
            return f"{prefix} {percentage:.1f}% - significant errors present."
        else:
            return f"{prefix} {percentage:.1f}% - very large errors, model needs work."

    def _explain_accuracy(self, accuracy: float) -> str:
        """Explain accuracy percentage"""
        if accuracy >= 95:
            return f"Model is correct {accuracy:.1f}% of the time - outstanding performance!"
        elif accuracy >= 85:
            return f"Model is correct {accuracy:.1f}% of the time - very good!"
        elif accuracy >= 75:
            return f"Model is correct {accuracy:.1f}% of the time - good but could improve."
        elif accuracy >= 65:
            return f"Model is correct {accuracy:.1f}% of the time - moderate, needs work."
        else:
            return f"Model is correct {accuracy:.1f}% of the time - poor performance, significant retraining needed."

    def _assign_performance_grade(self, accuracy: float, mae: float) -> str:
        """Assign a letter grade to model performance"""
        score = accuracy * 0.7 + (1 - min(mae, 1)) * 30  # Weighted score
        
        if score >= 90:
            return "A+ - Exceptional"
        elif score >= 80:
            return "A - Excellent"
        elif score >= 70:
            return "B - Good"
        elif score >= 60:
            return "C - Fair"
        elif score >= 50:
            return "D - Poor"
        else:
            return "F - Critical"

    def _generate_detailed_insights(self, predictions: List[float], actuals: List[float], errors: List[float]) -> Dict:
        """Generate detailed insights about model behavior"""
        
        errors = np.array(errors)
        
        # Find best and worst predictions
        abs_errors = np.abs(errors)
        best_idx = np.argmin(abs_errors)
        worst_idx = np.argmax(abs_errors)
        
        insights = {
            "best_prediction": {
                "accuracy": f"Error: {abs_errors[best_idx]:.4f}",
                "description": f"Most accurate prediction - predicted {predictions[best_idx]:.4f}, actual {actuals[best_idx]:.4f}"
            },
            "worst_prediction": {
                "accuracy": f"Error: {abs_errors[worst_idx]:.4f}",
                "description": f"Least accurate prediction - predicted {predictions[worst_idx]:.4f}, actual {actuals[worst_idx]:.4f}"
            },
            "bias_analysis": self._analyze_model_bias(predictions, actuals)
        }
        
        return insights

    def _analyze_model_bias(self, predictions: List[float], actuals: List[float]) -> str:
        """Analyze if model has systematic bias"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mean_error = np.mean(actuals - predictions)
        
        if abs(mean_error) < 0.02:
            return "✅ No bias detected - model predictions are balanced."
        elif mean_error > 0:
            return f"📊 Systematic underestimation - model tends to predict {abs(mean_error):.2%} too low on average."
        else:
            return f"📊 Systematic overestimation - model tends to predict {abs(mean_error):.2%} too high on average."

    # ============================================================================
    # VISUALIZATION HELPERS
    # ============================================================================

    def format_insights_for_display(self, insight) -> str:
        """Format any insight object for display"""
        if isinstance(insight, PredictionInsight):
            return self._format_prediction_insight(insight)
        elif isinstance(insight, ErrorInsight):
            return self._format_error_insight(insight)
        else:
            return str(insight)

    def _format_prediction_insight(self, insight: PredictionInsight) -> str:
        """Format prediction insight for display"""
        text = f"""
### 🔮 Prediction Analysis

**Prediction Value:** {insight.prediction:.4f}  
**Confidence Level:** {insight.confidence}

**What This Means:**  
{insight.user_friendly_description}

**Probability Explanation:**  
{insight.probability_explanation}

**Recent Trend:**  
{insight.trend_analysis}

**Recommendations:**
"""
        for rec in insight.recommendations:
            text += f"\n- {rec}"
        
        return text

    def _format_error_insight(self, insight: ErrorInsight) -> str:
        """Format error insight for display"""
        text = f"""
### 📊 Error Analysis

**Error Value:** {insight.error_value:.4f}  
**Error Percentage:** {insight.error_percentage:.2f}%  
**Severity:** {insight.severity}

**Description:**  
{insight.user_friendly_description}

**Possible Causes:**
"""
        for cause in insight.possible_causes:
            text += f"\n- {cause}"
        
        text += "\n\n**Improvement Suggestions:**\n"
        for suggestion in insight.improvement_suggestions:
            text += f"\n- {suggestion}"
        
        text += f"\n\n**Performance Context:**  \n{insight.comparison_context}"
        
        return text


# ============================================================================
# UTILITY FUNCTIONS FOR EASY USE
# ============================================================================

def create_interpreter_bot() -> InterpreterBot:
    """Factory function to create an interpreter bot instance"""
    return InterpreterBot()


def explain_prediction_simple(prediction: float) -> str:
    """Quick function to get simple explanation of a prediction"""
    bot = InterpreterBot()
    insight = bot.interpret_prediction(prediction)
    return insight.user_friendly_description


def explain_error_simple(predicted: float, actual: float) -> str:
    """Quick function to get simple explanation of an error"""
    bot = InterpreterBot()
    insight = bot.interpret_error(predicted, actual)
    return insight.user_friendly_description
