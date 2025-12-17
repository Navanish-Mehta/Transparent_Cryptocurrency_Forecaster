"""
Explainability utilities: SHAP for tree models and generic feature importance.
Also includes plotting helpers for actual vs predicted and confidence intervals.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
	import shap
	SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
	shap = None
	SHAP_AVAILABLE = False


sns.set(style="whitegrid")


def compute_shap_values(model, X: pd.DataFrame):
	"""Compute SHAP values for tree-based models."""
	if not SHAP_AVAILABLE:
		raise ImportError("shap is not installed. Install shap>=0.41.0")
	
	try:
		# Try TreeExplainer first (for XGBoost, RandomForest)
		explainer = shap.TreeExplainer(model)
		values = explainer.shap_values(X)
		# Handle case where SHAP returns list for multi-output
		if isinstance(values, list):
			values = values[0]
		return values
	except Exception as e:
		# Fallback to KernelExplainer for non-tree models
		try:
			explainer = shap.KernelExplainer(model.predict, X.iloc[:100])  # Sample for speed
			values = explainer.shap_values(X.iloc[:100])
			return values
		except Exception as e2:
			raise ImportError(f"SHAP computation failed: {e}. Fallback also failed: {e2}")


def plot_shap_summary(shap_values, X: pd.DataFrame):
	"""Plot SHAP summary plot."""
	if not SHAP_AVAILABLE:
		raise ImportError("shap is not installed.")
	
	try:
		# Handle different SHAP versions
		if hasattr(shap, 'summary_plot'):
			shap.summary_plot(shap_values, X, plot_type="bar", show=False)
		else:
			# Fallback for older versions
			shap.summary_plot(shap_values, X, show=False)
		plt.tight_layout()
		return plt.gcf()
	except Exception as e:
		# Create a simple feature importance plot as fallback
		plt.figure(figsize=(8, 4))
		mean_shap = np.abs(shap_values).mean(0)
		feature_names = X.columns
		order = np.argsort(mean_shap)[::-1]
		plt.bar(range(len(order)), mean_shap[order])
		plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=45, ha='right')
		plt.title("SHAP Feature Importance (Mean |SHAP|)")
		plt.tight_layout()
		return plt.gcf()


def plot_feature_importance(model, feature_names: list):
	"""Plot feature importance for models that support it."""
	if hasattr(model, "feature_importances_"):
		fi = model.feature_importances_
		order = np.argsort(fi)[::-1]
		plt.figure(figsize=(8, 4))
		
		# Ensure feature_names are strings and handle potential issues
		feature_names_str = [str(name) for name in feature_names]
		top_features = [feature_names_str[i] for i in order][:20]
		top_importance = fi[order][:20]
		
		plt.bar(range(len(top_features)), top_importance)
		plt.xticks(range(len(top_features)), top_features, rotation=45, ha="right")
		plt.title("Feature Importance (Top 20)")
		plt.tight_layout()
		return plt.gcf()
	else:
		raise ValueError("Model does not provide feature_importances_")


def plot_actual_vs_pred(y_true: pd.Series, y_pred: pd.Series, title: str = "Actual vs Predicted"):
	"""Plot actual vs predicted values."""
	y_true, y_pred = y_true.align(y_pred, join="inner")
	plt.figure(figsize=(10, 4))
	plt.plot(y_true.index, y_true.values, label="Actual", alpha=0.7)
	plt.plot(y_pred.index, y_pred.values, label="Predicted", alpha=0.7)
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	return plt.gcf()


def plot_forecast_with_ci(pred: pd.Series, ci: pd.DataFrame, title: str = "Forecast with CI"):
	"""Plot forecast with confidence intervals."""
	plt.figure(figsize=(10, 4))
	plt.plot(pred.index, pred.values, label="Forecast", linewidth=2)
	if ci is not None and {"lower", "upper"}.issubset(ci.columns):
		plt.fill_between(ci.index, ci["lower"], ci["upper"], color="orange", alpha=0.2, label="95% CI")
	plt.title(title)
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	return plt.gcf()