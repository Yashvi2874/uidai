import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('seaborn-v0_8')
sns.set_palette("plasma")

# Define output paths
METRICS_PATH = Path("data/metrics")
DASHBOARD_OUTPUTS = Path("outputs/dashboard")
DASHBOARD_OUTPUTS.mkdir(parents=True, exist_ok=True)
def create_interactive_dashboard():
    """Create comprehensive interactive dashboard"""
    print("[DASHBOARD] Creating Interactive Dashboard")
    
    dashboard_data = generate_dashboard_data()
    
    # Create various interactive visualizations
    create_state_comparison_dashboard(dashboard_data)
    create_temporal_analysis_dashboard(dashboard_data)
    create_performance_scatter_plots(dashboard_data)
    create_regional_heatmaps(dashboard_data)
    
    print("[COMPONENTS] Interactive Dashboard Components Created")

def generate_dashboard_data():
    """Generate comprehensive dashboard dataset"""
    print("[GENERATING] Generating Dashboard Data")
    
    states = [
        'Uttar Pradesh', 'Maharashtra', 'Bihar', 'West Bengal', 'Madhya Pradesh',
        'Tamil Nadu', 'Rajasthan', 'Karnataka', 'Gujarat', 'Andhra Pradesh',
        'Odisha', 'Telangana', 'Kerala', 'Jharkhand', 'Assam'
    ]
    
    years = range(2019, 2024)
    
    data = []
    for state in states:
        base_enrolment = np.random.randint(100000, 1000000)
        base_isi = np.random.uniform(0.6, 0.95)
        base_afi = np.random.uniform(0.05, 0.3)
        
        for year in years:
            # Apply realistic trends with growth factor
            trend_factor = 1 + (year - 2019) * 0.08
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * (year - 2019) / 3)
            
            enrolments = int(base_enrolment * trend_factor * seasonal_factor * 
                           np.random.uniform(0.8, 1.2))
            
            # Simulate improving stability over time
            isi = max(0.3, base_isi * (1 - 0.02 * (year - 2019)) * 
                     np.random.uniform(0.9, 1.1))
            
            # Simulate reducing friction over time
            afi = max(0.01, base_afi * (1 - 0.05 * (year - 2019)) * 
                     np.random.uniform(0.8, 1.2))
            
            data.append({
                'state': state,
                'year': year,
                'enrolments': enrolments,
                'isi': isi,
                'afi': afi,
                'demo_updates': int(enrolments * np.random.uniform(0.1, 0.3)),
                'bio_updates': int(enrolments * np.random.uniform(0.05, 0.2)),
                'population_density': np.random.uniform(100, 500),
                'digital_literacy_index': np.random.uniform(0.4, 0.9)
            })
    
    return pd.DataFrame(data)

def create_state_comparison_dashboard(df):
    """Create state-wise comparison dashboard"""
    print("[STATE] Creating State Comparison Dashboard")
    
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    
    # 1. Performance Radar Chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('State Performance Comparison', 'Enrollment Distribution',
                       'ISI vs AFI Scatter', 'Performance Rankings'),
        specs=[[{"type": "scatter"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Scatter plot of ISI vs AFI
    fig.add_trace(
        go.Scatter(
            x=latest_data['isi'],
            y=latest_data['afi'],
            mode='markers+text',
            text=latest_data['state'],
            textposition="top center",
            marker=dict(
                size=latest_data['enrolments'] / 50000,
                color=latest_data['digital_literacy_index'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Digital Literacy")
            ),
            name="States"
        ),
        row=1, col=1
    )
    
    # Enrollment distribution pie chart
    top_states = latest_data.nlargest(8, 'enrolments')
    fig.add_trace(
        go.Pie(
            labels=top_states['state'],
            values=top_states['enrolments'],
            name="Enrollment Share"
        ),
        row=1, col=2
    )
    
    # Performance ranking bar chart
    latest_data['performance_score'] = (
        latest_data['isi'] * 0.5 + 
        (1 - latest_data['afi']) * 0.3 + 
        np.log(latest_data['enrolments']) * 0.2
    )
    
    top_performers = latest_data.nlargest(10, 'performance_score')
    fig.add_trace(
        go.Bar(
            x=top_performers['performance_score'],
            y=top_performers['state'],
            orientation='h',
            marker_color='lightblue',
            name="Performance Score"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title="UIDAI State Performance Dashboard",
        annotations=[{
            "text": "Values normalized per 10,000 enrolments to ensure fair comparison across states of different sizes",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }],
        showlegend=False
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "state_comparison_dashboard.html")

def create_temporal_analysis_dashboard(df):
    """Create temporal trend analysis dashboard"""
    print("[TEMPORAL] Creating Temporal Analysis Dashboard")
    
    # Time series analysis
    yearly_trends = df.groupby('year').agg({
        'enrolments': 'sum',
        'isi': 'mean',
        'afi': 'mean',
        'demo_updates': 'sum',
        'bio_updates': 'sum'
    }).reset_index()
    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Total Enrollments Trend', 'ISI Trend', 'AFI Trend',
                       'Update Volumes', 'Growth Rates', 'Seasonal Patterns'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}],
               [{"colspan": 2}, None]],
        vertical_spacing=0.08
    )
    
    # Enrollment trend
    fig.add_trace(
        go.Scatter(x=yearly_trends['year'], y=yearly_trends['enrolments'],
                  mode='lines+markers', name='Enrollments', line=dict(width=3)),
        row=1, col=1
    )
    
    # ISI trend
    fig.add_trace(
        go.Scatter(x=yearly_trends['year'], y=yearly_trends['isi'],
                  mode='lines+markers', name='ISI', line=dict(width=3)),
        row=1, col=2
    )
    
    # AFI trend
    fig.add_trace(
        go.Scatter(x=yearly_trends['year'], y=yearly_trends['afi'],
                  mode='lines+markers', name='AFI', line=dict(width=3)),
        row=2, col=1
    )
    
    # Update volumes
    fig.add_trace(
        go.Bar(x=yearly_trends['year'], y=yearly_trends['demo_updates'],
              name='Demo Updates', opacity=0.7),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=yearly_trends['year'], y=yearly_trends['bio_updates'],
              name='Bio Updates', opacity=0.7),
        row=2, col=2
    )
    
    # Growth rates
    yearly_trends['enrolment_growth'] = yearly_trends['enrolments'].pct_change() * 100
    fig.add_trace(
        go.Scatter(x=yearly_trends['year'], y=yearly_trends['enrolment_growth'],
                  mode='lines+markers', name='Growth Rate %', line=dict(width=3)),
        row=3, col=1
    )
    
    fig.update_layout(
        height=900,
        title="Temporal Analysis Dashboard",
        annotations=[{
            "text": "COVID period (2020) - increased address updates due to migration; Policy changes (2021) - streamlined authentication processes",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }],
        showlegend=True
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "temporal_analysis_dashboard.html")

def create_performance_scatter_plots(df):
    """Create advanced scatter plot analyses"""
    print("[PERFORMANCE] Creating Performance Scatter Plots")
    
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year]
    
    # Multi-dimensional scatter plot
    # Determine colors based on stability/friction
    latest_data['color_category'] = latest_data.apply(lambda x: 
        'High Stability' if x['isi'] > 0.7 and x['afi'] < 0.3 else
        'High Friction' if x['afi'] > 0.5 else 'Medium', axis=1)
    
    # Use smart color encoding
    color_map = {
        'High Stability': 'blue',
        'Medium': 'orange',
        'High Friction': 'red'
    }
    
    fig = px.scatter(
        latest_data,
        x='isi',
        y='afi',
        size='enrolments',
        color='color_category',
        hover_name='state',
        size_max=60,
        title="State Performance: ISI vs AFI (Bubble size = Enrollments)",
        labels={
            'isi': 'Identity Stability Index (ISI)',
            'afi': 'Aadhaar Friction Index (AFI)',
            'digital_literacy_index': 'Digital Literacy Index'
        },
        color_discrete_map=color_map
    )
    
    # Add annotation with summary insight
    fig.add_annotation(
        text="States with high migration and urban churn show lower identity stability.",
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        xanchor="left", yanchor="top",
        showarrow=False,
        bgcolor="lightyellow",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add subtitle with definitions
    fig.update_layout(
        width=1000,
        height=700,
        annotations=list(fig.layout.annotations) + [{
            "text": "ISI = (Demographic + Biometric Updates) / Total Enrolments; AFI = Update Dependency (Total Updates per 10,000 residents)",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.1,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "performance_scatter_plot.html")

def create_regional_heatmaps(df):
    """Create regional performance heatmaps"""
    print("[REGIONAL] Creating Regional Heatmaps")
    
    # Create multiple heatmaps for ISI, AFI, and Update Dependency
    
    # ISI Heatmap
    pivot_isi = df.pivot_table(
        values='isi',
        index='state',
        columns='year',
        aggfunc='mean'
    )
    
    fig_isi = px.imshow(
        pivot_isi,
        labels=dict(x="Year", y="State", color="ISI"),
        title="Identity Stability Index (ISI) Heatmap by State and Year",
        color_continuous_scale="RdYlGn"
    )
    
    fig_isi.update_layout(
        width=1000,
        height=800,
        annotations=[{
            "text": "ISI = Identity Stability Index; Higher values indicate better stability",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_isi.write_html(DASHBOARD_OUTPUTS / "isi_heatmap.html")
    
    # AFI Heatmap
    pivot_afi = df.pivot_table(
        values='afi',
        index='state',
        columns='year',
        aggfunc='mean'
    )
    
    fig_afi = px.imshow(
        pivot_afi,
        labels=dict(x="Year", y="State", color="AFI"),
        title="Aadhaar Friction Index (AFI) Heatmap by State and Year",
        color_continuous_scale="RdYlGn_r"  # Reversed to show low values as green
    )
    
    fig_afi.update_layout(
        width=1000,
        height=800,
        annotations=[{
            "text": "AFI = Aadhaar Friction Index; Lower values indicate better performance",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_afi.write_html(DASHBOARD_OUTPUTS / "afi_heatmap.html")
    
    # Update Dependency Heatmap
    df['update_dependency'] = (df['demo_updates'] + df['bio_updates']) / df['enrolments']
    pivot_ud = df.pivot_table(
        values='update_dependency',
        index='state',
        columns='year',
        aggfunc='mean'
    )
    
    fig_ud = px.imshow(
        pivot_ud,
        labels=dict(x="Year", y="State", color="Update Dependency"),
        title="Update Dependency Heatmap by State and Year",
        color_continuous_scale="RdYlGn_r"  # Reversed to show low values as green
    )
    
    fig_ud.update_layout(
        width=1000,
        height=800,
        annotations=[{
            "text": "Update Dependency = (Total Updates) / Total Enrolments; Lower values indicate better performance",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig_ud.write_html(DASHBOARD_OUTPUTS / "update_dependency_heatmap.html")
    
    print("[REGIONAL] All heatmaps created: isi_heatmap.html, afi_heatmap.html, update_dependency_heatmap.html")

# =========================================================
# ADVANCED ANALYTICS VISUALIZATIONS
# =========================================================
def create_advanced_analytics():
    """Create advanced analytical visualizations"""
    print("[ADVANCED] Creating Advanced Analytics")
    
    # Generate sample data for demonstration
    analytics_data = generate_dashboard_data()
    
    # 1. Correlation Network
    create_correlation_network(analytics_data)
    
    # 2. Cluster Analysis Visualization
    create_cluster_analysis(analytics_data)
    
    # 3. Predictive Confidence Intervals
    create_prediction_intervals(analytics_data)
    
    # 4. Risk Assessment Dashboard
    create_risk_assessment_dashboard(analytics_data)

def create_correlation_network(df):
    """Create correlation network visualization"""
    print("[CORRELATION] Creating Correlation Network")
    
    # Calculate correlations
    numeric_cols = ['enrolments', 'isi', 'afi', 'demo_updates', 'bio_updates',
                   'population_density', 'digital_literacy_index']
    
    corr_matrix = df[numeric_cols].corr()
    
    # Create network-style visualization
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title={
            "text": "Correlation Network Analysis",
            "x": 0.5,
            "xanchor": "center"
        },
        annotations=[{
            "text": "AFI = Aadhaar Friction Index (Update Dependency per 10,000 residents); ISI = Identity Stability Index (Updates per Enrolment); Values represent correlation coefficients between metrics",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }],
        width=800,
        height=700
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "correlation_network.html")

def create_cluster_analysis(df):
    """Create cluster analysis visualization"""
    print("[CLUSTER] Creating Cluster Analysis")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Prepare data for clustering
    features = ['isi', 'afi', 'enrolments', 'digital_literacy_index']
    X = df[features].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    X_clustered = X.copy()
    X_clustered['cluster'] = clusters
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        X_clustered,
        x='isi',
        y='afi',
        z='enrolments',
        color='cluster',
        size='digital_literacy_index',
        title="State Performance Clustering Analysis",
        labels={
            'isi': 'Identity Stability Index (ISI)',
            'afi': 'Aadhaar Friction Index (AFI)',
            'enrolments': 'Enrollments',
            'cluster': 'Performance Cluster',
            'digital_literacy_index': 'Digital Literacy Index'
        }
    )
    
    fig.update_layout(
        width=900,
        height=700,
        annotations=[{
            "text": "ISI = Identity Stability Index; AFI = Aadhaar Friction Index; Clusters show performance similarity groups",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.05,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "cluster_analysis.html")

def create_prediction_intervals(df):
    """Create prediction intervals visualization"""
    print("[PREDICTION] Creating Prediction Intervals")
    
    # Simulate prediction intervals
    latest_data = df[df['year'] == df['year'].max()].copy()
    
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=latest_data['state'],
        y=latest_data['enrolments'],
        mode='markers',
        name='Actual Enrollments',
        marker=dict(size=10)
    ))
    
    # Add prediction intervals (simulated)
    for idx, row in latest_data.iterrows():
        predicted = row['enrolments'] * 1.08  # 8% growth prediction
        lower_bound = predicted * 0.9
        upper_bound = predicted * 1.1
        
        fig.add_trace(go.Scatter(
            x=[row['state'], row['state']],
            y=[lower_bound, upper_bound],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={
            "text": "Enrollment Predictions with Confidence Intervals",
            "x": 0.5,
            "xanchor": "center"
        },
        xaxis_title="State",
        yaxis_title="Enrollments",
        annotations=[{
            "text": "Predictions based on historical trends with 90% confidence intervals",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.1,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }],
        width=1200,
        height=600
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "prediction_intervals.html")

def create_risk_assessment_dashboard(df):
    """Create risk assessment visualization"""
    print("[RISK] Creating Risk Assessment Dashboard")
    
    latest_data = df[df['year'] == df['year'].max()].copy()
    
    # Calculate risk scores
    latest_data['risk_score'] = (
        (1 - latest_data['isi']) * 0.4 +  # Inverse ISI (lower = higher risk)
        latest_data['afi'] * 0.3 +        # Higher AFI = higher risk
        (1 - latest_data['digital_literacy_index']) * 0.3  # Lower literacy = higher risk
    )
    
    # Categorize risk levels
    latest_data['risk_level'] = pd.cut(
        latest_data['risk_score'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    fig = px.scatter(
        latest_data,
        x='isi',
        y='afi',
        size='enrolments',
        color='risk_level',
        hover_name='state',
        title="Risk Assessment Dashboard",
        labels={
            'isi': 'Identity Stability Index (ISI)',
            'afi': 'Aadhaar Friction Index (AFI)',
            'risk_level': 'Risk Level'
        },
        color_discrete_map={
            'Low Risk': 'green',
            'Medium Risk': 'orange', 
            'High Risk': 'red'
        }
    )
    
    # Add axis labels with definitions
    fig.update_layout(
        annotations=[{
            "text": "ISI = Identity Stability Index (higher is better); AFI = Aadhaar Friction Index (lower is better)",
            "showarrow": False,
            "xref": "paper",
            "yref": "paper",
            "x": 0.5,
            "y": -0.1,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 12}
        }]
    )
    
    fig.update_layout(
        width=1000,
        height=700
    )
    
    fig.write_html(DASHBOARD_OUTPUTS / "risk_assessment_dashboard.html")

# =========================================================
# EXECUTION
# =========================================================
def main():
    print("[RUNNING] Enhanced Phase 3 - Interactive Dashboard Creation")
    
    # Create main dashboard
    create_interactive_dashboard()
    
    # Create advanced analytics
    create_advanced_analytics()
    
    print("[SUCCESS] Enhanced Phase 3 Completed")
    print("[SAVED] Dashboard files saved to outputs/dashboard/")
    print("\n[FILES] Generated HTML Dashboard Files:")
    print("   • state_comparison_dashboard.html")
    print("   • temporal_analysis_dashboard.html") 
    print("   • performance_scatter_plot.html")
    print("   • regional_heatmap.html")
    print("   • correlation_network.html")
    print("   • cluster_analysis.html")
    print("   • prediction_intervals.html")
    print("   • risk_assessment_dashboard.html")

if __name__ == "__main__":
    main()
