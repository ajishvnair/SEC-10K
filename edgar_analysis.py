"""
EDGAR Analysis System - Demo Version with Anomaly Visualization
Author: Ajish V Nair
Date: January 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class EDGARAnalyzer:
    def __init__(self):
        """Initialize with sample data for demonstration"""
        self.data = self._create_sample_data()
        
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample financial data for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='Q')
        companies = ['Company_A', 'Company_B', 'Company_C']
        
        data = []
        for company in companies:
            base_revenue = np.random.uniform(1000, 5000)
            growth_rate = np.random.uniform(0.02, 0.05)
            
            for i, date in enumerate(dates):
                revenue = base_revenue * (1 + growth_rate) ** i
                profit = revenue * np.random.uniform(0.1, 0.2)
                assets = revenue * np.random.uniform(2, 3)
                debt = assets * np.random.uniform(0.3, 0.5)
                
                data.append({
                    'company': company,
                    'date': date,
                    'revenue': revenue,
                    'profit': profit,
                    'assets': assets,
                    'debt': debt,
                    'profit_margin': profit / revenue,
                    'debt_ratio': debt / assets
                })
        
        return pd.DataFrame(data)

    def detect_anomalies(self) -> pd.DataFrame:
        """Detect anomalies in financial metrics"""
        numeric_cols = ['revenue', 'profit', 'debt_ratio', 'profit_margin']
        X = self.data[numeric_cols]
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Use K-means for anomaly detection
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        # Calculate distances to cluster centers
        distances = np.min(kmeans.transform(X_pca), axis=1)
        
        # Flag anomalies (top 5% of distances)
        threshold = np.percentile(distances, 95)
        anomalies = distances > threshold
        
        # Create DataFrame with results
        anomaly_df = pd.DataFrame({
            'company': self.data['company'],
            'date': self.data['date'],
            'revenue': self.data['revenue'],
            'profit': self.data['profit'],
            'debt_ratio': self.data['debt_ratio'],
            'profit_margin': self.data['profit_margin'],
            'is_anomaly': anomalies,
            'anomaly_score': distances
        })
        
        return anomaly_df

    def visualize_insights(self):
        """Generate visualizations for key insights with anomaly highlights"""
        sns.set_style("whitegrid")
        
        # Get anomaly detection results
        anomaly_data = self.detect_anomalies()
        
        # Create a figure with 2x3 subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3)
        
        # 1. Revenue Trends with Anomalies
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=self.data, x='date', y='revenue', hue='company', ax=ax1)
        
        # Highlight anomalies
        anomalies = anomaly_data[anomaly_data['is_anomaly']]
        ax1.scatter(anomalies['date'], anomalies['revenue'], 
                   color='red', marker='*', s=200, label='Anomaly',
                   zorder=5)
        ax1.set_title('Revenue Trends (★ = Anomaly)')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Profit vs Revenue with Anomalies
        ax2 = fig.add_subplot(gs[0, 1])
        sns.scatterplot(data=self.data, x='revenue', y='profit', 
                       hue='company', style='company', ax=ax2)
        
        # Highlight anomalies
        ax2.scatter(anomalies['revenue'], anomalies['profit'],
                   color='red', marker='*', s=200, label='Anomaly',
                   zorder=5)
        ax2.set_title('Revenue vs Profit (★ = Anomaly)')
        
        # 3. Anomaly Scores Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sns.histplot(data=anomaly_data, x='anomaly_score', bins=30, ax=ax3)
        ax3.axvline(x=np.percentile(anomaly_data['anomaly_score'], 95),
                    color='red', linestyle='--', label='Anomaly Threshold')
        ax3.set_title('Distribution of Anomaly Scores')
        ax3.legend()
        
        # 4. Profit Margin Over Time with Anomalies
        ax4 = fig.add_subplot(gs[1, 0])
        sns.lineplot(data=self.data, x='date', y='profit_margin', 
                    hue='company', ax=ax4)
        
        # Highlight anomalies
        ax4.scatter(anomalies['date'], anomalies['profit_margin'],
                   color='red', marker='*', s=200, label='Anomaly',
                   zorder=5)
        ax4.set_title('Profit Margin Trends (★ = Anomaly)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Debt Ratio Analysis with Anomalies
        ax5 = fig.add_subplot(gs[1, 1:])
        sns.boxplot(data=self.data, x='company', y='debt_ratio', ax=ax5)
        
        # Add individual points for anomalies
        sns.swarmplot(data=anomalies, x='company', y='debt_ratio',
                     color='red', marker='*', size=10, ax=ax5)
        ax5.set_title('Debt Ratio Distribution by Company (★ = Anomaly)')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed anomaly information
        print("\nDetailed Anomaly Information:")
        print("-" * 80)
        for _, row in anomalies.iterrows():
            print(f"Company: {row['company']}")
            print(f"Date: {row['date'].strftime('%Y-%m-%d')}")
            print(f"Revenue: ${row['revenue']:,.2f}")
            print(f"Profit Margin: {row['profit_margin']:.2%}")
            print(f"Debt Ratio: {row['debt_ratio']:.2f}")
            print(f"Anomaly Score: {row['anomaly_score']:.2f}")
            print("-" * 80)

    def generate_report(self) -> Dict:
        """Generate a comprehensive analysis report"""
        anomalies = self.detect_anomalies()
        
        # Calculate key statistics
        avg_profit_margin = self.data.groupby('company')['profit_margin'].mean()
        avg_debt_ratio = self.data.groupby('company')['debt_ratio'].mean()
        
        report = {
            'summary_metrics': {
                'average_profit_margins': avg_profit_margin.to_dict(),
                'average_debt_ratios': avg_debt_ratio.to_dict(),
                'total_companies': self.data['company'].nunique(),
                'date_range': {
                    'start': self.data['date'].min(),
                    'end': self.data['date'].max()
                }
            },
            'anomalies_detected': anomalies['is_anomaly'].sum(),
            'company_rankings': avg_profit_margin.sort_values(ascending=False).index.tolist(),
            'anomalies_detail': anomalies[anomalies['is_anomaly']].to_dict('records')
        }
        
        return report

def main():
    print("Initializing EDGAR Analyzer...")
    analyzer = EDGARAnalyzer()
    
    print("\nGenerating analysis report...")
    report = analyzer.generate_report()
    
    print("\nAnalysis Report:")
    print("-" * 50)
    print(f"Analysis Period: {report['summary_metrics']['date_range']['start'].date()} to "
          f"{report['summary_metrics']['date_range']['end'].date()}")
    print(f"Companies Analyzed: {report['summary_metrics']['total_companies']}")
    
    print("\nCompany Rankings (by profit margin):")
    for i, company in enumerate(report['company_rankings'], 1):
        margin = report['summary_metrics']['average_profit_margins'][company]
        print(f"{i}. {company}: {margin:.2%}")
    
    print(f"\nAnomalies Detected: {report['anomalies_detected']}")
    
    print("\nGenerating visualizations with anomaly highlights...")
    analyzer.visualize_insights()

if __name__ == "__main__":
    main()
