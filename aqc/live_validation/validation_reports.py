import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, List

class ValidationReports:
    """Generates CSV reports and plots from validation results."""
    
    def __init__(self, output_dir: str = "reports", plots_dir: str = "plots"):
        self.output_dir = output_dir
        self.plots_dir = plots_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def export_alpha_decay_report(self, data: List[Dict[str, Any]], filename: str = "alpha_decay_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_signal_stability_report(self, data: List[Dict[str, Any]], filename: str = "signal_stability_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_forecast_stability_report(self, data: List[Dict[str, Any]], filename: str = "forecast_stability_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_regime_drift_report(self, data: List[Dict[str, Any]], filename: str = "regime_drift_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_execution_validation_report(self, data: List[Dict[str, Any]], filename: str = "execution_validation_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_strategy_health_report(self, data: List[Dict[str, Any]], filename: str = "strategy_health_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_governance_report(self, data: List[Dict[str, Any]], filename: str = "governance_report.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def export_retraining_recommendations(self, data: List[Dict[str, Any]], filename: str = "retraining_recommendations.csv"):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, filename), index=False)
        
    def plot_alpha_decay_curve(self, timeseries_data: List[Dict[str, Any]], filename: str = "alpha_decay_curve.png"):
        df = pd.DataFrame(timeseries_data)
        if df.empty or 'timestamp' not in df or 'sharpe_decay' not in df:
            return
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='timestamp', y='sharpe_decay', marker='o')
        plt.title('Alpha Decay (Sharpe) Over Time')
        plt.ylabel('Decay %')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
        
    def plot_health_score_timeline(self, timeseries_data: List[Dict[str, Any]], filename: str = "health_score_timeline.png"):
        df = pd.DataFrame(timeseries_data)
        if df.empty or 'timestamp' not in df or 'overall_score' not in df:
            return
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='timestamp', y='overall_score', marker='o')
        plt.axhline(y=70, color='r', linestyle='--', label='Warning Threshold')
        plt.title('Strategy Health Score Timeline')
        plt.ylabel('Health Score (0-100)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()

    def plot_regime_drift_heatmap(self, expected: Dict[str, float], observed: Dict[str, float], filename: str = "regime_drift_heatmap.png"):
        all_keys = list(set(expected.keys()).union(set(observed.keys())))
        if not all_keys:
            return
            
        expected_arr = [expected.get(k, 0.0) for k in all_keys]
        observed_arr = [observed.get(k, 0.0) for k in all_keys]
        
        df = pd.DataFrame({'Expected': expected_arr, 'Observed': observed_arr}, index=all_keys)
        
        plt.figure(figsize=(8, 5))
        sns.heatmap(df.T, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Regime Distribution Drift")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()
