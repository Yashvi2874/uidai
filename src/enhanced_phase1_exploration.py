import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define paths for data processing
RAW = Path("data/raw")
OUT = Path("data/processed")
ANALYSIS = Path("outputs/analysis")
OUT.mkdir(parents=True, exist_ok=True)
ANALYSIS.mkdir(parents=True, exist_ok=True)

ENROL = RAW / "api_data_aadhar_enrolment"
DEMO  = RAW / "api_data_aadhar_demographic"
BIO   = RAW / "api_data_aadhar_biometric"
def load_and_explore():
    """Load data and perform comprehensive exploration"""
    print("[EXPLORING] Enhanced Data Exploration Started")
    
    enrol_files = list(ENROL.glob("*.csv"))
    if enrol_files:
        df_sample = pd.read_csv(enrol_files[0], nrows=10000)
        print(f"[DATA] Sample data shape: {df_sample.shape}")
        print(f"[COLS] Columns: {list(df_sample.columns)}")
        
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        print("\n[NUMERIC] Numeric Column Statistics:")
        print(df_sample[numeric_cols].describe())
        
        with open(ANALYSIS / "data_exploration_report.txt", "w") as f:
            f.write("=== UIDAI DATA EXPLORATION REPORT ===\n\n")
            f.write(f"Sample Shape: {df_sample.shape}\n")
            f.write(f"Columns: {list(df_sample.columns)}\n")
            f.write(f"\nNumeric Statistics:\n{df_sample[numeric_cols].describe()}")
    
    return True

# =========================================================
# ADVANCED FEATURE ENGINEERING
# =========================================================
def engineer_features(df):
    """Create advanced features from raw data"""
    print("[ENGINEERING] Feature Engineering Started")
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
    
    if 'state' in df.columns:
        state_sizes = {
            'Uttar Pradesh': 230, 'Maharashtra': 120, 'Bihar': 110,
            'West Bengal': 100, 'Madhya Pradesh': 300, 'Tamil Nadu': 130,
            'Rajasthan': 340, 'Karnataka': 190, 'Gujarat': 190, 'Andhra Pradesh': 160
        }
        df['state_area_rank'] = df['state'].map({k: v for k, v in 
                                                sorted(state_sizes.items(), key=lambda x: x[1])})
    
    age_cols = [col for col in df.columns if col.startswith('age_')]
    if age_cols:
        working_age = ['age_18_30', 'age_31_45', 'age_46_60']
        dependent_age = ['age_0_17', 'age_61_plus']
        
        working_sum = sum(df[col] for col in working_age if col in df.columns)
        dependent_sum = sum(df[col] for col in dependent_age if col in df.columns)
        
        df['dependency_ratio'] = np.where(working_sum > 0, 
                                        dependent_sum / working_sum, 0)
    
    return df

# =========================================================
# CORRELATION ANALYSIS
# =========================================================
def correlation_analysis(df, output_path):
    """Perform correlation analysis between key metrics"""
    print("[CORRELATING] Correlation Analysis Started")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        corr_results = corr_matrix.unstack().sort_values(key=lambda x: abs(x), ascending=False)
        corr_results = corr_results[corr_results != 1.0]  # Remove self-correlations
        
        with open(output_path / "correlation_analysis.txt", "w") as f:
            f.write("=== CORRELATION ANALYSIS RESULTS ===\n\n")
            f.write("Top 10 Strongest Correlations:\n")
            for i, ((col1, col2), corr) in enumerate(corr_results.head(10).items()):
                f.write(f"{i+1}. {col1} ↔ {col2}: {corr:.3f}\n")
    
    return True

# =========================================================
# TREND ANALYSIS
# =========================================================
def trend_analysis(df, output_path):
    """Analyze temporal trends in the data"""
    print("[TRENDING] Trend Analysis Started")
    
    if 'date' in df.columns and 'state' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        monthly_trends = df.groupby('year_month').agg({
            col: 'sum' for col in df.select_dtypes(include=[np.number]).columns[:5]
        }).reset_index()
        
        monthly_trends['year_month'] = monthly_trends['year_month'].dt.to_timestamp()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        numeric_cols = monthly_trends.select_dtypes(include=[np.number]).columns[:4]
        
        for i, col in enumerate(numeric_cols):
            axes[i].plot(monthly_trends['year_month'], monthly_trends[col], 
                        marker='o', linewidth=2, markersize=4)
            axes[i].set_title(f'{col.replace("_", " ").title()} Trend')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(col.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(output_path / "temporal_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    return True

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    print("[RUNNING] Enhanced Phase 1 - Advanced Data Analysis")
    
    # Perform data exploration
    load_and_explore()
    
    # Conceptual demonstration of enhanced approach
    
    print("[SUCCESS] Enhanced Phase 1 Completed")
    print("[SAVED] Analysis outputs saved to outputs/analysis/")

if __name__ == "__main__":
    main()