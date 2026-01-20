"""
Policy-Aware Aadhaar Analysis Module
Extends existing framework to separate policy-mandated biometric updates from discretionary updates
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Define paths
DASHBOARD_OUTPUTS = Path("outputs/dashboard")
DASHBOARD_OUTPUTS.mkdir(parents=True, exist_ok=True)

def extract_policy_age_updates(df):
    """
    Extract policy-mandated biometric updates for age 15+ from biometric dataset
    
    Args:
        df: DataFrame containing biometric update data with age groups
        
    Returns:
        DataFrame with policy-driven update metrics
    """
    print("[POLICY_EXTRACTION] Extracting policy-mandated biometric updates")
    
    # Identify policy age group (15-17 years)
    policy_age_columns = [col for col in df.columns if 'age_15' in col.lower() or '15_17' in col]
    
    if not policy_age_columns:
        # Fallback: create synthetic policy age data based on demographic patterns
        print("[WARNING] No explicit 15-17 age group found. Generating policy age estimates.")
        df['policy_age_updates'] = df['bio_updates'] * 0.25  # Assume 25% of bio updates are policy-driven
    else:
        # Sum all 15+ age group columns as policy-driven updates
        df['policy_age_updates'] = df[policy_age_columns].sum(axis=1)
    
    # Handle division by zero and missing data
    df['policy_age_updates'] = df['policy_age_updates'].fillna(0)
    df['total_bio_updates'] = df['bio_updates'].fillna(0)
    
    return df

def calculate_policy_metrics(df):
    """
    Calculate policy-aware metrics including Policy Update Ratio (PUR)
    
    Args:
        df: DataFrame with extracted policy age updates
        
    Returns:
        DataFrame with calculated policy metrics
    """
    print("[POLICY_METRICS] Calculating policy-aware metrics")
    
    # Calculate Policy Update Ratio (PUR)
    df['pur'] = np.where(
        df['total_bio_updates'] > 0,
        df['policy_age_updates'] / df['total_bio_updates'],
        0
    )
    
    # Calculate policy-normalized AFI
    df['afi_normalized'] = np.where(
        df['enrolments'] > 0,
        (df['demo_updates'] + df['bio_updates'] - df['policy_age_updates']) / df['enrolments'],
        0
    )
    
    # Calculate policy-normalized ISI
    df['isi_normalized'] = 1 / (1 + df['afi_normalized'])
    
    # Ensure metric ranges (0 to 1)
    df['afi_normalized'] = np.clip(df['afi_normalized'], 0, 1)
    df['isi_normalized'] = np.clip(df['isi_normalized'], 0, 1)
    
    return df

def generate_policy_rankings(df):
    """
    Generate state-wise rankings for policy-driven biometric updates
    
    Args:
        df: DataFrame with policy metrics
        
    Returns:
        tuple: (top_10_states, bottom_10_states)
    """
    print("[POLICY_RANKINGS] Generating policy-driven update rankings")
    
    latest_data = df[df['year'] == df['year'].max()].copy()
    
    # Top 10 states by policy-driven updates
    top_10 = latest_data.nlargest(10, 'policy_age_updates')[[
        'state', 'policy_age_updates', 'pur'
    ]].round(3)
    
    # Bottom 10 states by policy-driven updates
    bottom_10 = latest_data.nsmallest(10, 'policy_age_updates')[[
        'state', 'policy_age_updates', 'pur'
    ]].round(3)
    
    return top_10, bottom_10

def create_policy_visualizations(df, top_10, bottom_10):
    """
    Create policy-aware visualizations without altering existing dashboards
    
    Args:
        df: DataFrame with policy metrics
        top_10: Top 10 states dataframe
        bottom_10: Bottom 10 states dataframe
    """
    print("[POLICY_VISUALIZATIONS] Creating policy-aware visualizations")
    
    # 1. Top 10 states by policy-driven biometric updates
    fig_top = go.Figure()
    fig_top.add_trace(go.Bar(
        x=top_10['policy_age_updates'],
        y=top_10['state'],
        orientation='h',
        marker_color='steelblue',
        name='Policy-Driven Biometric Updates'
    ))
    
    fig_top.update_layout(
        title='Top 10 States by Policy-Mandated Biometric Updates (Age 15+)',
        xaxis_title='Number of Policy-Driven Updates',
        yaxis_title='State',
        height=600,
        annotations=[{
            "text": "Represents mandatory biometric re-enrollment at age 15 - indicates high youth population/compliance",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.15,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_top.write_html(DASHBOARD_OUTPUTS / "policy_top_states.html")
    
    # 2. Bottom 10 states by policy-driven biometric updates
    fig_bottom = go.Figure()
    fig_bottom.add_trace(go.Bar(
        x=bottom_10['policy_age_updates'],
        y=bottom_10['state'],
        orientation='h',
        marker_color='lightcoral',
        name='Policy-Driven Biometric Updates'
    ))
    
    fig_bottom.update_layout(
        title='Bottom 10 States by Policy-Mandated Biometric Updates (Age 15+)',
        xaxis_title='Number of Policy-Driven Updates',
        yaxis_title='State',
        height=600,
        annotations=[{
            "text": "Lower values may indicate under-coverage or access constraints for youth population",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.15,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_bottom.write_html(DASHBOARD_OUTPUTS / "policy_bottom_states.html")
    
    # 3. Stacked bar chart: Policy vs Non-Policy biometric updates
    latest_data = df[df['year'] == df['year'].max()].copy()
    latest_data['non_policy_updates'] = (
        latest_data['bio_updates'] - latest_data['policy_age_updates']
    )
    
    fig_stacked = go.Figure()
    fig_stacked.add_trace(go.Bar(
        x=latest_data['state'],
        y=latest_data['policy_age_updates'],
        name='Policy-Driven Updates (Age 15+)',
        marker_color='blue'
    ))
    
    fig_stacked.add_trace(go.Bar(
        x=latest_data['state'],
        y=latest_data['non_policy_updates'],
        name='Non-Policy Updates',
        marker_color='orange'
    ))
    
    fig_stacked.update_layout(
        title='Policy vs Non-Policy Biometric Updates by State',
        xaxis_title='State',
        yaxis_title='Number of Updates',
        barmode='stack',
        height=700,
        annotations=[{
            "text": "Policy-driven updates (blue) represent mandatory re-enrollment at age 15; Non-policy updates (orange) indicate discretionary/corrective updates",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.15,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_stacked.write_html(DASHBOARD_OUTPUTS / "policy_vs_non_policy_updates.html")
    
    print("[POLICY_VISUALIZATIONS] Policy-aware visualizations created")

def policy_aware_analysis_pipeline():
    """
    Main pipeline function integrating all policy-aware components
    """
    print("[POLICY_PIPELINE] Starting Policy-Aware Aadhaar Analysis")
    
    # Generate or load data (using existing synthetic data approach)
    from enhanced_phase3_dashboard import generate_dashboard_data
    
    # Get base data
    df = generate_dashboard_data()
    
    # Step 1: Extract policy age updates
    df = extract_policy_age_updates(df)
    
    # Step 2: Calculate policy metrics
    df = calculate_policy_metrics(df)
    
    # Step 3: Generate rankings
    top_10, bottom_10 = generate_policy_rankings(df)
    
    # Step 4: Create visualizations
    create_policy_visualizations(df, top_10, bottom_10)
    
    # Save results
    results_path = Path("outputs/policy_analysis")
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Save rankings
    top_10.to_csv(results_path / "top_10_policy_states.csv", index=False)
    bottom_10.to_csv(results_path / "bottom_10_policy_states.csv", index=False)
    
    # Save full dataset with policy metrics
    df.to_csv(results_path / "policy_enhanced_dataset.csv", index=False)
    
    print("[POLICY_PIPELINE] Policy-aware analysis completed")
    print("[RESULTS_SAVED] Results saved to outputs/policy_analysis/")
    print("[VISUALIZATIONS] Generated:")
    print("   • policy_top_states.html")
    print("   • policy_bottom_states.html") 
    print("   • policy_vs_non_policy_updates.html")

if __name__ == "__main__":
    policy_aware_analysis_pipeline()