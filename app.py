import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient

st.set_page_config(page_title="ML Footprint", layout="wide")

st.title("Machine Learning Model Footprint Dashboard")
st.markdown("Compare carbon/energy cost vs accuracy for ML models")

@st.cache_data
def load_data():
    """Load MLflow experiment data."""
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        client = MlflowClient()
        experiment = client.get_experiment_by_name("green_ai_comparison")
        
        if not experiment:
            return pd.DataFrame()
        
        runs = client.search_runs(experiment.experiment_id)
        
        data = []
        for run in runs:
            m = run.data.metrics
            p = run.data.params
            
            data.append({
                'Model': p.get('model', 'Unknown'),
                'Dataset': p.get('dataset', 'Unknown'),
                'Accuracy': m.get('accuracy', m.get('r2_score', 0)),
                'F1 Score': m.get('f1_score'),
                'Time (s)': m.get('training_time_sec', 0),
                'CO₂ (kg)': m.get('co2_kg', 0),
                'Energy (kWh)': m.get('energy_kwh', 0),
            })
        
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No data found. Run training first:")
    st.code("python train.py")
    st.stop()

# Filters
st.sidebar.header("Filters")
datasets = st.sidebar.multiselect("Datasets", df['Dataset'].unique(), df['Dataset'].unique())
models = st.sidebar.multiselect("Models", df['Model'].unique(), df['Model'].unique())

df_filtered = df[df['Dataset'].isin(datasets) & df['Model'].isin(models)]

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Experiments", len(df_filtered))
col2.metric("Total CO₂", f"{df_filtered['CO₂ (kg)'].sum():.6f} kg")
col3.metric("Total Energy", f"{df_filtered['Energy (kWh)'].sum():.4f} kWh")
col4.metric("Avg Accuracy", f"{df_filtered['Accuracy'].mean():.3f}")

st.markdown("---")

# Pareto Chart
st.subheader("Accuracy vs CO₂ Emissions (Pareto)")
fig = px.scatter(
    df_filtered,
    x='CO₂ (kg)',
    y='Accuracy',
    color='Model',
    size='Time (s)',
    hover_data=['Dataset'],
    title="Lower CO₂ + Higher Accuracy = Better"
)
st.plotly_chart(fig, use_container_width=True)

st.info("Best models are in the **top-left** corner (high accuracy, low emissions)")

# Side by side charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Time")
    fig_time = px.bar(df_filtered, x='Model', y='Time (s)', color='Dataset')
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("CO₂ Emissions")
    fig_co2 = px.bar(df_filtered, x='Model', y='CO₂ (kg)', color='Dataset')
    st.plotly_chart(fig_co2, use_container_width=True)

# Data table
st.subheader("Results")
st.dataframe(df_filtered.style.format({
    'Accuracy': '{:.4f}',
    'Time (s)': '{:.2f}',
    'CO₂ (kg)': '{:.6f}',
    'Energy (kWh)': '{:.4f}'
}), use_container_width=True)
```

## Step 5: Create `.gitignore`
```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.ipynb_checkpoints/

# MLflow
mlruns/

# CodeCarbon
emissions.csv
*.log

# Environment
.env
.venv/
venv/
env/

# OS
.DS_Store
Thumbs.db