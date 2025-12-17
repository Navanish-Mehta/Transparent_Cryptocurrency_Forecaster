"""
Streamlit frontend for Transparent Cryptocurrency Price Forecaster.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import traceback

from data_fetcher import prepare_dataset, SUPPORTED_SYMBOLS
from models import (
	fit_arima, forecast_arima, evaluate_arima,
	fit_prophet, forecast_prophet, evaluate_prophet,
	fit_xgboost, forecast_xgboost, evaluate_xgb,
)
from explainability import (
	compute_shap_values, plot_shap_summary, plot_feature_importance,
	plot_actual_vs_pred, plot_forecast_with_ci,
)

st.set_page_config(page_title="Transparent Crypto Forecaster", layout="wide")

st.title("Transparent Cryptocurrency Price Forecaster")

# Add some styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
	st.header("Configuration")
	symbol = st.selectbox("Cryptocurrency", options=list(SUPPORTED_SYMBOLS.keys()), index=0, format_func=lambda s: f"{SUPPORTED_SYMBOLS[s]} ({s})")
	horizon = st.slider("Forecast days", 7, 30, 7)
	lags = st.slider("Lag features (for XGBoost)", 5, 30, 10)
	model_choice = st.selectbox("Model", ["ARIMA", "Prophet", "XGBoost"], index=2)
	
	st.markdown("---")
	st.markdown("**Tips:**")
	st.markdown("• XGBoost provides SHAP explainability")
	st.markdown("• ARIMA/Prophet show confidence intervals")
	st.markdown("• Adjust lags for better ML performance")

# Main content
try:
	st.write("📊 Fetching and preparing data...")
	with st.spinner("Loading cryptocurrency data..."):
		raw_df, dataset = prepare_dataset(symbol=symbol, lags=lags, test_size=0.2)
	
	# Helper function to get the correct close column
	def get_close_column(df):
		if "Close" in df.columns:
			return "Close"
		else:
			# Try to find the close column with different case variations
			for col in df.columns:
				if col.lower() == "close":
					return col
			raise ValueError("Could not find 'Close' column in the data")
	
	# Get the close column name
	close_col = get_close_column(raw_df)
	
	st.success(f"✅ Data loaded successfully! {len(raw_df)} records from {raw_df.index[0].strftime('%Y-%m-%d')} to {raw_df.index[-1].strftime('%Y-%m-%d')}")
	
	# Show data summary
	col1, col2, col3 = st.columns(3)
	with col1:
		current_price = float(raw_df[close_col].iloc[-1].item())
		st.metric("Current Price", f"${current_price:,.2f}")
	with col2:
		# Calculate 24h change from the original data
		if len(raw_df) >= 2:
			daily_return = (raw_df[close_col].iloc[-1] - raw_df[close_col].iloc[-2]) / raw_df[close_col].iloc[-2]
			return_value = float(daily_return.item())
			st.metric("24h Change", f"{return_value*100:.2f}%")
		else:
			st.metric("24h Change", "N/A")
	with col3:
		st.metric("Data Points", len(raw_df))

	st.subheader("📈 OHLCV Data (Recent 365 days)")
	st.line_chart(raw_df[[close_col]].tail(365))

	# Train/forecast per selected model
	if model_choice == "ARIMA":
		st.subheader("🔮 ARIMA Forecast")
		with st.spinner("Training ARIMA model..."):
			try:
				model = fit_arima(raw_df[close_col])
				st.info(f"ARIMA{model.order} model fitted successfully")
				
				with st.spinner("Generating forecast..."):
					pred, ci = forecast_arima(model, steps=horizon)
				
				# Show forecast metrics
				col1, col2 = st.columns(2)
				with col1:
					forecast_start = float(pred.iloc[0].item())
					st.metric("Forecast Start", f"${forecast_start:,.2f}")
				with col2:
					forecast_end = float(pred.iloc[-1].item())
					st.metric("Forecast End", f"${forecast_end:,.2f}")
				
				st.pyplot(plot_forecast_with_ci(pred, ci, title=f"ARIMA {symbol} {horizon}d Forecast"))
				
			except Exception as e:
				st.error(f"ARIMA model failed: {str(e)}")
				st.code(traceback.format_exc())
				
	elif model_choice == "Prophet":
		st.subheader("🔮 Prophet Forecast")
		with st.spinner("Training Prophet model..."):
			try:
				model = fit_prophet(raw_df[close_col])
				st.success("Prophet model fitted successfully")
				
				with st.spinner("Generating forecast..."):
					pred, ci = forecast_prophet(model, steps=horizon)
				
				# Show forecast metrics
				col1, col2 = st.columns(2)
				with col1:
					forecast_start = float(pred.iloc[0].item())
					st.metric("Forecast Start", f"${forecast_start:,.2f}")
				with col2:
					forecast_end = float(pred.iloc[-1].item())
					st.metric("Forecast End", f"${forecast_end:,.2f}")
				
				st.pyplot(plot_forecast_with_ci(pred, ci, title=f"Prophet {symbol} {horizon}d Forecast"))
				
			except Exception as e:
				st.error(f"Prophet model failed: {str(e)}")
				st.code(traceback.format_exc())
				
	else:  # XGBoost
		st.subheader("🤖 XGBoost Forecast and Explainability")
		with st.spinner("Training XGBoost model..."):
			try:
				xgb_model = fit_xgboost(dataset.X_train, dataset.y_train)
				st.success("XGBoost model fitted successfully")
				
				# Test predictions
				with st.spinner("Generating test predictions..."):
					y_pred_test = forecast_xgboost(xgb_model, dataset.X_test)
				
				# Calculate and display metrics
				metrics = evaluate_xgb(dataset.y_test, y_pred_test)
				st.markdown("**Test Set Performance:**")
				col1, col2, col3 = st.columns(3)
				with col1:
					rmse_value = float(metrics['RMSE'])
					st.metric("RMSE", f"${rmse_value:,.2f}")
				with col2:
					mae_value = float(metrics['MAE'])
					st.metric("MAE", f"${mae_value:,.2f}")
				with col3:
					mape_value = float(metrics['MAPE'])
					st.metric("MAPE", f"{mape_value:.2f}%")
				
				# Actual vs Predicted plot
				st.pyplot(plot_actual_vs_pred(dataset.y_test, y_pred_test, title="Test Actual vs Predicted (XGBoost)"))
				
				# Future forecast
				with st.spinner("Generating future forecast..."):
					future_pred = forecast_xgboost(xgb_model, dataset.X_test, steps=horizon)
				
				st.markdown("**Future Forecast:**")
				st.line_chart(future_pred)
				
				# Explainability
				st.markdown("**Model Explainability:**")
				
				# SHAP values
				try:
					with st.spinner("Computing SHAP values..."):
						shap_values = compute_shap_values(xgb_model, dataset.X_test)
					st.pyplot(plot_shap_summary(shap_values, dataset.X_test))
				except Exception as e:
					st.warning(f"SHAP analysis not available: {str(e)}")
					st.info("This might be due to SHAP version compatibility or model type")
				
				# Feature importance
				try:
					st.pyplot(plot_feature_importance(xgb_model, dataset.feature_names))
				except Exception as e:
					st.warning(f"Feature importance not available: {str(e)}")
					
			except Exception as e:
				st.error(f"XGBoost model failed: {str(e)}")
				st.code(traceback.format_exc())

except Exception as e:
	st.error("❌ An error occurred while processing the request")
	st.error(f"Error: {str(e)}")
	st.code(traceback.format_exc())
	st.info("Please check your internet connection and try again. If the problem persists, try a different cryptocurrency or model.")

# Footer
st.markdown("---")
st.caption("🎓 Academic Project: Transparent Cryptocurrency Price Forecaster with Explainable AI Made By Navanish✨")
st.caption("⚠️ Disclaimer: This is for educational purposes only. Financial predictions are uncertain and should not be used for investment decisions.")