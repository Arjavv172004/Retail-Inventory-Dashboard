import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import time
from PIL import Image
import io
import base64
import streamlit_nested_layout
import uuid
import random
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Retail Inventory Optimization System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .stButton button {
            background-color: #1e88e5;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 500;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #0d47a1;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        h1, h2, h3 {
            color: #ffffff;
            font-weight: 600;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 10px 24px;
            border-radius: 4px 4px 0px 0px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #e3f2fd;
            border-bottom: 2px solid #1e88e5;
        }
        
        .metric-card {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-label {
            font-size: 1rem;
            color: #666;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 600;
            color: #333;
        }
        
        .custom-info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #1e88e5;
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        
        .stDataFrame {
            border-radius: L8px;
            overflow: hidden;
        }
        
        /* Animation for progress */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .loading-animation {
            animation: pulse 1.5s infinite;
        }
        
        /* Code block styling */
        code {
            font-family: 'Roboto Mono', monospace;
            background-color: #f8f9fa;
            padding: 4px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        /* Dashboard panels */
        .dashboard-container {
            background-color: #f9fafb;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }
        
        /* Consistent colors for agents */
        .store-agent { color: #4caf50; }
        .warehouse-agent { color: #2196f3; }
        .supplier-agent { color: #ff9800; }
        .customer-agent { color: #9c27b0; }
    </style>
""", unsafe_allow_html=True)

# Helper function to display metrics
def display_metric(title, value, delta=None, prefix="", suffix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {f'<div style="font-size: 0.9rem; color: {"green" if delta >= 0 else "red"};">{"+" if delta >= 0 else ""}{delta}%</div>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)

# Helper function for custom progress animation
def progress_animation(progress_text="Operation in progress", duration=3):
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{progress_text} - {i+1}%")
        time.sleep(duration/100)
    
    status_text.text("Complete!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

# Class for simulated multi-agent system
class RetailAgent:
    def __init__(self, agent_type, name, data=None):
        self.agent_type = agent_type
        self.name = name
        self.data = data
        self.id = str(uuid.uuid4())[:8]
        self.state = "idle"
        self.messages = []
        self.predictions = {}
    
    def send_message(self, to_agent, message_content):
        """Simulate sending a message to another agent"""
        message = {
            "from": self.name,
            "to": to_agent.name,
            "content": message_content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": str(uuid.uuid4())[:8]
        }
        self.messages.append(message)
        to_agent.messages.append(message)
        return message
    
    def process_data(self, data):
        """Process data according to agent type"""
        if self.agent_type == "store":
            # Store agent processes inventory data
            result = self._process_store_data(data)
        elif self.agent_type == "warehouse":
            # Warehouse agent processes supply chain data
            result = self._process_warehouse_data(data)
        elif self.agent_type == "supplier":
            # Supplier agent processes lead times and costs
            result = self._process_supplier_data(data)
        elif self.agent_type == "customer":
            # Customer agent processes demand and preferences
            result = self._process_customer_data(data)
        else:
            result = None
        
        self.state = "processed"
        return result
    
    def _process_store_data(self, data):
        """Store agent data processing logic"""
        if 'Stock Levels' in data.columns:
            low_stock_items = data[data['Stock Levels'] < data['Reorder Point']]
            return {
                "low_stock_count": len(low_stock_items),
                "low_stock_items": low_stock_items,
                "avg_stock_level": data['Stock Levels'].mean(),
                "stockout_risk": len(low_stock_items) / len(data) if len(data) > 0 else 0
            }
        return {"status": "No inventory data available"}
    
    def _process_warehouse_data(self, data):
        """Warehouse agent data processing logic"""
        if 'Warehouse Capacity' in data.columns and 'Stock Levels' in data.columns:
            capacity_usage = data['Stock Levels'].sum() / data['Warehouse Capacity'].sum() if data['Warehouse Capacity'].sum() > 0 else 0
            return {
                "capacity_usage": capacity_usage,
                "avg_lead_time": data['Supplier Lead Time (days)'].mean() if 'Supplier Lead Time (days)' in data.columns else 0,
                "total_stock": data['Stock Levels'].sum()
            }
        return {"status": "No warehouse data available"}
    
    def _process_supplier_data(self, data):
        """Supplier agent data processing logic"""
        if 'Supplier Lead Time (days)' in data.columns:
            return {
                "avg_lead_time": data['Supplier Lead Time (days)'].mean(),
                "max_lead_time": data['Supplier Lead Time (days)'].max(),
                "order_fulfillment_time": data['Order Fulfillment Time (days)'].mean() if 'Order Fulfillment Time (days)' in data.columns else 0
            }
        return {"status": "No supplier data available"}
    
    def _process_customer_data(self, data):
        """Customer agent data processing logic"""
        if 'Sales Quantity' in data.columns:
            return {
                "avg_sales": data['Sales Quantity'].mean(),
                "total_sales": data['Sales Quantity'].sum(),
                "sales_trend": data.groupby('Date')['Sales Quantity'].sum().pct_change().mean() if 'Date' in data.columns else 0
            }
        return {"status": "No customer data available"}
    
    def predict(self, data, target_column, features=None):
        """Make predictions based on historical data"""
        if features is None:
            # Automatically select numeric features
            features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove the target from features if it exists
            if target_column in features:
                features.remove(target_column)
        
        if len(features) == 0 or target_column not in data.columns:
            return {"error": "Insufficient data for prediction"}
        
        try:
            X = data[features]
            y = data[target_column]
            
            # Handle NaN values
            X = X.fillna(X.mean())
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a RandomForest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store prediction results
            self.predictions[target_column] = {
                "model": model,
                "features": features,
                "metrics": {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2
                },
                "feature_importance": dict(zip(features, model.feature_importances_))
            }
            
            return self.predictions[target_column]
        
        except Exception as e:
            return {"error": str(e)}
    
    def recommend_actions(self, data):
        """Generate recommendations based on data analysis"""
        recommendations = []
        
        if self.agent_type == "store" and 'Stock Levels' in data.columns and 'Reorder Point' in data.columns:
            # Identify items that need reordering
            reorder_items = data[data['Stock Levels'] < data['Reorder Point']]
            for _, item in reorder_items.iterrows():
                recommendations.append({
                    "action": "Reorder",
                    "product_id": item['Product ID'],
                    "current_stock": item['Stock Levels'],
                    "reorder_point": item['Reorder Point'],
                    "priority": "High" if item['Stock Levels'] < item['Reorder Point'] * 0.5 else "Medium"
                })
        
        elif self.agent_type == "warehouse" and 'Stock Levels' in data.columns and 'Warehouse Capacity' in data.columns:
            # Optimize warehouse space
            avg_capacity = data['Warehouse Capacity'].mean()
            high_stock = data[data['Stock Levels'] > 0.8 * avg_capacity]
            for _, item in high_stock.iterrows():
                recommendations.append({
                    "action": "Redistribute",
                    "product_id": item['Product ID'],
                    "current_stock": item['Stock Levels'],
                    "capacity": item['Warehouse Capacity'],
                    "priority": "Medium"
                })
        
        elif self.agent_type == "supplier" and 'Supplier Lead Time (days)' in data.columns:
            # Supplier recommendations
            long_lead_time = data[data['Supplier Lead Time (days)'] > data['Supplier Lead Time (days)'].mean() * 1.2]
            for _, item in long_lead_time.iterrows():
                recommendations.append({
                    "action": "Optimize Lead Time",
                    "product_id": item['Product ID'],
                    "lead_time": item['Supplier Lead Time (days)'],
                    "avg_lead_time": data['Supplier Lead Time (days)'].mean(),
                    "priority": "Medium"
                })
        
        elif self.agent_type == "customer" and 'Price' in data.columns and 'Sales Quantity' in data.columns:
            # Price optimization
            low_sales = data[data['Sales Quantity'] < data['Sales Quantity'].mean() * 0.8]
            for _, item in low_sales.iterrows():
                recommendations.append({
                    "action": "Price Adjustment",
                    "product_id": item['Product ID'],
                    "current_price": item['Price'],
                    "sales_quantity": item['Sales Quantity'],
                    "avg_sales": data['Sales Quantity'].mean(),
                    "priority": "Medium"
                })
        
        return recommendations

# Load and preprocess data
# Load and preprocess data
@st.cache_data
def load_data():
    try:
        # Load the actual CSV files
        demand_df = pd.read_csv("demand_forecasting.csv")
        inventory_df = pd.read_csv("inventory_monitoring.csv")
        pricing_df = pd.read_csv("pricing_optimization.csv")
        
        # Convert Date column to datetime if it exists
        if 'Date' in demand_df.columns:
            demand_df['Date'] = pd.to_datetime(demand_df['Date'])
        
        # Convert Expiry Date to datetime if it exists
        if 'Expiry Date' in inventory_df.columns:
            inventory_df['Expiry Date'] = pd.to_datetime(inventory_df['Expiry Date'])
        
        # Basic data cleaning
        # Fill missing numerical values with column means
        for df in [demand_df, inventory_df, pricing_df]:
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(df[col].mean())
        
        return demand_df, inventory_df, pricing_df
        
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Using simulated data instead.")
        # Fall back to simulated data if files aren't found
        return generate_simulated_data()

# Function to generate simulated data as fallback
def generate_simulated_data():
    # The existing simulation code goes here
    # (keep all the existing simulation code from the original load_data function)
    
    # Demand forecasting data
    date_range = pd.date_range(start='2023-01-01', periods=365, freq='D')
    product_ids = range(1, 21)
    store_ids = range(1, 6)
    
    # ... rest of the simulated data generation code ...
    
    return demand_df, inventory_df, pricing_df

# Load the data
demand_df, inventory_df, pricing_df = load_data()

# Initialize agents
store_agent = RetailAgent("store", "Store Agent", inventory_df)
warehouse_agent = RetailAgent("warehouse", "Warehouse Agent", inventory_df)
supplier_agent = RetailAgent("supplier", "Supplier Agent", inventory_df)
customer_agent = RetailAgent("customer", "Customer Agent", demand_df)

agents = [store_agent, warehouse_agent, supplier_agent, customer_agent]

# Function to merge data for analysis
def merge_datasets(demand_df, inventory_df, pricing_df):
    # Merge all datasets on Product ID and Store ID
    merged_df = demand_df.merge(inventory_df, on=['Product ID', 'Store ID'], how='inner')
    merged_df = merged_df.merge(pricing_df, on=['Product ID', 'Store ID'], how='inner')
    return merged_df

# Create merged dataset
merged_data = merge_datasets(demand_df, inventory_df, pricing_df)

# Main app structure
def main():
    # Sidebar for navigation and controls
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/warehouse.png", width=80)
        st.title("AI Supply Chain Management by Team Codifier")
        st.markdown("---")
        
        menu = st.radio(
            "Navigation",
            ["Dashboard", "Demand Forecasting", "Inventory Management", "Pricing Optimization", "Multi-Agent System", "Settings"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("### Filters")
        
        # Add filters
        selected_store = st.multiselect("Store", options=sorted(inventory_df['Store ID'].unique()), default=[1, 2])
        
        date_min = demand_df['Date'].min().date()
        date_max = demand_df['Date'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(date_max - timedelta(days=30), date_max),
            min_value=date_min,
            max_value=date_max
        )
        
        # Filter data based on selections
        if len(selected_store) > 0:
            filtered_inventory = inventory_df[inventory_df['Store ID'].isin(selected_store)]
            filtered_demand = demand_df[
                (demand_df['Store ID'].isin(selected_store)) & 
                (demand_df['Date'].dt.date >= date_range[0]) & 
                (demand_df['Date'].dt.date <= date_range[1])
            ]
            filtered_pricing = pricing_df[pricing_df['Store ID'].isin(selected_store)]
        else:
            filtered_inventory = inventory_df
            filtered_demand = demand_df[
                (demand_df['Date'].dt.date >= date_range[0]) & 
                (demand_df['Date'].dt.date <= date_range[1])
            ]
            filtered_pricing = pricing_df
        
        # Run simulation button
        if st.button("Run Optimization", key="run_sim"):
            with st.spinner("Running optimization..."):
                progress_animation("Analyzing data", 2)
                st.success("Optimization complete!")
        
        st.markdown("---")
        st.markdown("Supply Chain Management by Team Codifier")

    # Main content area
    if menu == "Dashboard":
        display_dashboard(filtered_demand, filtered_inventory, filtered_pricing)
    elif menu == "Demand Forecasting":
        display_demand_forecasting(filtered_demand)
    elif menu == "Inventory Management":
        display_inventory_management(filtered_inventory)
    elif menu == "Pricing Optimization":
        display_pricing_optimization(filtered_pricing)
    elif menu == "Multi-Agent System":
        display_multi_agent_system(agents, filtered_demand, filtered_inventory, filtered_pricing)
    elif menu == "Settings":
        display_settings()

# Dashboard page
# Multi-Agent System page (continuation from your code)
def display_multi_agent_system(agents, demand_df, inventory_df, pricing_df):
    # (Your existing code for this function remains)
    
    # Agent decisions and recommendations
    st.markdown("### AI Agent Recommendations")
    
    # Get recommendations from each agent
    store_recs = agents[0].recommend_actions(inventory_df)
    warehouse_recs = agents[1].recommend_actions(inventory_df)
    
    # Display recommendations in expandable sections
    with st.expander("Store Agent Recommendations"):
        for rec in store_recs:
            st.markdown(f"""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #4caf50; color: black;'>
                <strong>{rec['action'] if 'action' in rec else 'Recommendation'}</strong><br>
                {rec.get('description', f"Product ID: {rec.get('product_id', 'N/A')}")}<br>
                <small>Priority: {rec.get('priority', 'Medium')} | Impact: {rec.get('impact', 'Efficiency')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander("Warehouse Agent Recommendations"):
        for rec in warehouse_recs:
            st.markdown(f"""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #2196f3;'>
                <strong>{rec['action'] if 'action' in rec else 'Recommendation'}</strong><br>
                {rec.get('description', f"Product ID: {rec.get('product_id', 'N/A')}")}<br>
                <small>Priority: {rec.get('priority', 'Medium')} | Impact: {rec.get('impact', 'Efficiency')}</small>
            </div>
            """, unsafe_allow_html=True)

    # Rest of your function remains the same
    
    # Agent performance metrics
    st.markdown("### Agent Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a simulated performance chart
        performance_labels = ['Forecast Accuracy', 'Response Time', 'Decision Quality', 'Adaptation']
        store_perf = [85, 92, 78, 88]
        warehouse_perf = [90, 75, 82, 79]
        supplier_perf = [70, 85, 80, 75]
        customer_perf = [95, 90, 85, 92]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=store_perf,
            theta=performance_labels,
            fill='toself',
            name='Store Agent',
            line_color='#4caf50'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=warehouse_perf,
            theta=performance_labels,
            fill='toself',
            name='Warehouse Agent',
            line_color='#2196f3'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=supplier_perf,
            theta=performance_labels,
            fill='toself',
            name='Supplier Agent',
            line_color='#ff9800'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=customer_perf,
            theta=performance_labels,
            fill='toself',
            name='Customer Agent',
            line_color='#9c27b0'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create a simulated outcome improvement chart
        outcome_labels = ['Stockout Reduction', 'Cost Savings', 'Revenue Increase', 'Customer Satisfaction']
        outcome_values = [32, 18, 25, 15]
        
        fig = px.bar(
            x=outcome_labels,
            y=outcome_values,
            color=outcome_values,
            color_continuous_scale='Viridis',
            labels={'x': 'Outcome', 'y': 'Improvement (%)'},
            text=outcome_values
        )
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        
        fig.update_layout(
            title="System Improvements",
            xaxis_title=None,
            yaxis_title="Improvement (%)",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent learning and adaptation section
    st.markdown("### Agent Learning & Adaptation")
    
    # Simulated learning chart
    learning_data = pd.DataFrame({
        'Day': list(range(1, 31)),
        'Prediction Error': [20 - i * 0.5 + random.uniform(-2, 2) for i in range(30)],
        'Action Efficiency': [70 + i * 0.8 + random.uniform(-3, 3) for i in range(30)]
    })
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=learning_data['Day'],
            y=learning_data['Prediction Error'],
            mode='lines',
            name='Prediction Error (%)',
            line=dict(color='#f44336', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=learning_data['Day'],
            y=learning_data['Action Efficiency'],
            mode='lines',
            name='Action Efficiency (%)',
            line=dict(color='#4caf50', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Days",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Prediction Error (%)", secondary_y=False)
    fig.update_yaxes(title_text="Action Efficiency (%)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# Settings page
def display_settings():
    st.markdown("<h1 style='text-align: center;'>Settings</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff;'>Configure system parameters and preferences</p>", unsafe_allow_html=True)
    
    # Create tabs for different settings
    tab1, tab2, tab3 = st.tabs(["System Settings", "Agent Configuration", "Data Sources"])
    
    with tab1:
        st.markdown("### System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Theme", ["Light", "Dark", "System Default"], index=0)
            st.selectbox("Update Frequency", ["Real-time", "Hourly", "Daily", "Weekly"], index=1)
            st.selectbox("Default View", ["Dashboard", "Demand Forecasting", "Inventory Management", "Pricing Optimization"], index=0)
        
        with col2:
            st.checkbox("Enable notifications", value=True)
            st.checkbox("Show tooltips", value=True)
            st.checkbox("Auto-refresh data", value=True)
        
        st.slider("Data retention period (days)", min_value=30, max_value=365, value=90, step=30)
        
    with tab2:
        st.markdown("### Agent Configuration")
        
        # Agent activation status
        st.markdown("#### Agent Activation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Store Agent", value=True)
            st.checkbox("Warehouse Agent", value=True)
        
        with col2:
            st.checkbox("Supplier Agent", value=True)
            st.checkbox("Customer Agent", value=True)
        
        # Agent parameters
        st.markdown("#### Agent Parameters")
        
        agent_type = st.selectbox("Configure Agent", ["Store Agent", "Warehouse Agent", "Supplier Agent", "Customer Agent"])
        
        if agent_type == "Store Agent":
            st.slider("Inventory Threshold Alert (%)", min_value=10, max_value=50, value=20)
            st.slider("Pricing Adjustment Window (days)", min_value=1, max_value=30, value=7)
            st.slider("Forecast Confidence Level (%)", min_value=70, max_value=99, value=90)
        elif agent_type == "Warehouse Agent":
            st.slider("Restock Trigger Level (%)", min_value=20, max_value=60, value=40)
            st.slider("Order Batch Size Optimization", min_value=1, max_value=10, value=5)
            st.slider("Inter-store Transfer Threshold", min_value=10, max_value=50, value=30)
        elif agent_type == "Supplier Agent":
            st.slider("Lead Time Sensitivity", min_value=1, max_value=10, value=7)
            st.slider("Emergency Order Threshold", min_value=1, max_value=10, value=3)
            st.slider("Bulk Discount Threshold", min_value=10, max_value=100, value=50)
        else:  # Customer Agent
            st.slider("Price Sensitivity", min_value=1, max_value=10, value=6)
            st.slider("Trend Detection Window (days)", min_value=7, max_value=90, value=30)
            st.slider("Review Impact Factor", min_value=1, max_value=10, value=7)
        
    with tab3:
        st.markdown("### Data Sources")
        
        # Database connection
        st.markdown("#### Database Connection")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Database Host", value="localhost")
            st.text_input("Database Name", value="retail_inventory_db")
        
        with col2:
            st.text_input("Username", value="admin")
            st.text_input("Password", type="password", value="•••••••••")
        
        st.button("Test Connection")
        
        # Data import/export
        st.markdown("#### Data Import/Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.file_uploader("Import Data", accept_multiple_files=True)
        
        with col2:
            st.selectbox("Export Format", ["CSV", "Excel", "JSON", "SQL"])
            st.button("Export Data")
        
        # API Integration
        st.markdown("#### API Integration")
        st.text_input("API Endpoint", value="https://api.example.com/v1/retail-data")
        st.text_input("API Key", type="password", value="•••••••••••••••••••••••••••")
        st.checkbox("Enable API Synchronization", value=False)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

# Helper function for metrics display
def display_metric(label, value, delta=None, suffix=""):
    if delta is not None:
        st.metric(
            label=label,
            value=f"{value}{suffix}",
            delta=f"{delta}{suffix if not isinstance(delta, float) else '%'}"
        )
    else:
        st.metric(
            label=label,
            value=f"{value}{suffix}"
        )

# Helper function for progress animation
def progress_animation(task_label, delay_seconds):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        status_text.text(f"{task_label}... ({i}%)")
        time.sleep(delay_seconds / 100)

# Agent class definition
class Agent:
    def __init__(self, name, role, icon, color):
        self.name = name
        self.role = role
        self.icon = icon
        self.color = color
        self.messages = []
    
    def send_message(self, to_agent, content):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add message to own log
        self.messages.append({
            "from": self.name,
            "to": to_agent.name,
            "content": content,
            "timestamp": timestamp
        })
        
        # Add message to recipient's log
        to_agent.messages.append({
            "from": self.name,
            "to": to_agent.name,
            "content": content,
            "timestamp": timestamp
        })
    
    def recommend_actions(self, df):
        # Simulate recommendations based on agent role
        recommendations = []
        
        if self.role == "Store":
            # Find low stock items
            low_stock = df[df['Stock Levels'] < df['Reorder Point']]
            
            if not low_stock.empty:
                recommendations.append({
                    "title": "Reorder Low Stock Products",
                    "description": f"Reorder {len(low_stock)} products below reorder threshold.",
                    "priority": "High",
                    "impact": "Prevent stockouts"
                })
            
            # Find items with high stock
            high_stock = df[df['Stock Levels'] > 0.8 * df['Warehouse Capacity']]
            
            if not high_stock.empty:
                recommendations.append({
                    "title": "Promote Overstocked Items",
                    "description": f"Create promotions for {len(high_stock)} overstocked products.",
                    "priority": "Medium",
                    "impact": "Reduce holding costs"
                })
            
            # Add generic recommendations
            recommendations.append({
                "title": "Optimize Store Layout",
                "description": "Rearrange store layout based on customer traffic patterns.",
                "priority": "Low",
                "impact": "Increase sales conversion"
            })
            
        elif self.role == "Warehouse":
            # Find items with long lead times
            long_lead = df[df['Supplier Lead Time (days)'] > 7]
            
            if not long_lead.empty:
                recommendations.append({
                    "title": "Adjust Reorder Points",
                    "description": f"Increase reorder points for {len(long_lead)} products with long lead times.",
                    "priority": "Medium",
                    "impact": "Maintain buffer stock"
                })
            
            # Find items with frequent stockouts
            stockout_prone = df[df['Stockout Frequency'] > 0.1]
            
            if not stockout_prone.empty:
                recommendations.append({
                    "title": "Increase Safety Stock",
                    "description": f"Add safety stock for {len(stockout_prone)} stockout-prone products.",
                    "priority": "High",
                    "impact": "Avoid lost sales"
                })
            
            # Add generic recommendations
            recommendations.append({
                "title": "Optimize Warehouse Space",
                "description": "Reorganize warehouse layout for faster picking and packing.",
                "priority": "Medium",
                "impact": "Improve efficiency"
            })
        
        # Return top 3 recommendations
        return recommendations[:3]
    
# Demand Forecasting page
def display_demand_forecasting(demand_df):
    st.markdown("<h1 style='text-align: center;'>Demand Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff;'>Predict future product demand with AI</p>", unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Demand Prediction Model")
        
        # Add model settings in the sidebar
        st.sidebar.markdown("### Forecast Settings")
        forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
        confidence_level = st.sidebar.slider("Confidence Level (%)", 80, 99, 95)
        
        # Product selection
        products = sorted(demand_df['Product ID'].unique())
        selected_product = st.selectbox("Select Product for Forecast", products)
        
        # Filter data for selected product
        product_data = demand_df[demand_df['Product ID'] == selected_product]
        
        if not product_data.empty:
    # Group by date and calculate daily sales
            daily_sales = product_data.groupby('Date')['Sales Quantity'].sum().reset_index()

            if daily_sales['Sales Quantity'].sum() == 0:
                st.warning("Sales quantity is zero for the selected product. Cannot generate forecast.")
            else:
        # Generate future dates for forecast
                last_date = daily_sales['Date'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

        # Compute average and standard deviation
                

# Ensure date is datetime type and sorted
                daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
                daily_sales = daily_sales.sort_values('Date')

# Set Date as index for time series modeling
                ts = daily_sales.set_index('Date')['Sales Quantity']

# Fit ARIMA model
                try:
                    model = ARIMA(ts, order=(1, 1, 1))  # You can tune the (p,d,q) values
                    model_fit = model.fit()
    
    # Forecast
                    forecast_result = model_fit.get_forecast(steps=forecast_days)
                    forecast = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int(alpha=1 - confidence_level / 100)

    # Prepare forecast DataFrame
                    forecast_df = pd.DataFrame({
                        'Date': pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days),
                        'Forecast': forecast.values,
                        'Lower Bound': conf_int.iloc[:, 0].values,
                        'Upper Bound': conf_int.iloc[:, 1].values
                    })

                except Exception as e:
                    st.error(f"ARIMA Model failed to fit. Error: {e}")
                    return


        # Plotting and forecast statistics continues...

            
            # Plot historical data and forecast
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=daily_sales['Date'],
                y=daily_sales['Sales Quantity'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#1e88e5')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff9800', dash='dash')
            ))
            
            # Add uncertainty bands
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                y=forecast_df['Upper Bound'].tolist() + forecast_df['Lower Bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 152, 0, 0.2)',
                line=dict(color='rgba(255, 152, 0, 0)'),
                name=f'{confidence_level}% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"Demand Forecast for Product ID: {selected_product}",
                xaxis_title="Date",
                yaxis_title="Sales Quantity",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            st.markdown("### Forecast Statistics")
            
            forecast_stats = {
                "Mean Forecast": f"{np.mean(forecast):.1f} units/day",
                "Total Forecasted Sales": f"{np.sum(forecast):.0f} units",
                "Peak Demand Day": f"Day {np.argmax(forecast) + 1}",
                "Peak Demand": f"{np.max(forecast):.1f} units"
            }
            
            # Display forecast statistics in a nicer format
            col1, col2 = st.columns(2)
            for i, (stat, value) in enumerate(forecast_stats.items()):
                if i % 2 == 0:
                    col1.metric(stat, value)
                else:
                    col2.metric(stat, value)
        else:
            st.warning("No data available for the selected product.")
    
    with col2:
        st.markdown("### Accuracy Metrics")
        
        # Simulate model accuracy metrics
        metrics = {
            "MAPE": f"{np.random.uniform(5, 15):.1f}%",
            "MAE": f"{np.random.uniform(2, 8):.1f} units",
            "RMSE": f"{np.random.uniform(3, 10):.1f} units",
            "R²": f"{np.random.uniform(0.7, 0.95):.2f}"
        }
        
        for metric, value in metrics.items():
            st.metric(metric, value)
        
        # Feature importance chart
        st.markdown("### Feature Importance")
        
        features = ['Recent Sales Trend', 'Seasonality', 'Price', 'Promotions', 'Day of Week', 'Holidays']
        importance = [0.35, 0.25, 0.18, 0.12, 0.06, 0.04]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='Blues',
            text=[f"{x:.0%}" for x in importance]
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            xaxis_range=[0, max(importance) * 1.1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns section
    st.markdown("### Seasonal Patterns")
    
    # Create random seasonal data
    if 'Date' in demand_df.columns:
        demand_df['Month'] = demand_df['Date'].dt.month
        demand_df['Day of Week'] = demand_df['Date'].dt.dayofweek
        
        # Monthly pattern
        monthly_sales = demand_df.groupby('Month')['Sales Quantity'].mean().reset_index()
        monthly_sales['Month'] = monthly_sales['Month'].apply(lambda x: pd.Timestamp(2023, x, 1).strftime('%b'))
        
        # Daily pattern
        daily_sales = demand_df.groupby('Day of Week')['Sales Quantity'].mean().reset_index()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales['Day'] = daily_sales['Day of Week'].apply(lambda x: days[x])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly pattern chart
            fig = px.line(
                monthly_sales,
                x='Month',
                y='Sales Quantity',
                markers=True,
                title="Monthly Sales Pattern",
                labels={'Sales Quantity': 'Avg. Sales', 'Month': ''}
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Daily pattern chart
            fig = px.bar(
                daily_sales,
                x='Day',
                y='Sales Quantity',
                title="Day of Week Pattern",
                labels={'Sales Quantity': 'Avg. Sales', 'Day': ''},
                color='Sales Quantity',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Inventory Management page
def display_inventory_management(inventory_df):
    st.markdown("<h1 style='text-align: center;'>Inventory Management</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff;'>Monitor and optimize inventory levels</p>", unsafe_allow_html=True)
    
    # Create tabs for different inventory views
    tabs = st.tabs(["Overview", "Stock Levels", "Reorder Planning", "Expiry Management"])
    
    with tabs[0]:  # Overview tab
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stock = inventory_df['Stock Levels'].sum()
            display_metric("Total Stock", f"{total_stock:,.0f} units")
        
        with col2:
            low_stock_count = len(inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']])
            display_metric("Low Stock Items", f"{low_stock_count} products", 
                          delta=f"{low_stock_count/len(inventory_df)*100:.1f}%" if len(inventory_df) > 0 else "0%")
        
        with col3:
            avg_stock_days = inventory_df['Stock Levels'].sum() / inventory_df['Sales Rate (daily)'].sum() if 'Sales Rate (daily)' in inventory_df.columns and inventory_df['Sales Rate (daily)'].sum() > 0 else 0
            display_metric("Avg. Stock Days", f"{avg_stock_days:.1f} days")
        
        with col4:
            stockout_risk = len(inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]) / len(inventory_df) if len(inventory_df) > 0 else 0
            display_metric("Stockout Risk", f"{stockout_risk*100:.1f}%", 
                          delta=f"{-5.2}" if stockout_risk < 0.1 else f"{3.8}")
        
        # Inventory health overview
        st.markdown("### Inventory Health")
        
        # Create buckets for inventory health
        inventory_df['Health Status'] = 'Normal'
        inventory_df.loc[inventory_df['Stock Levels'] < inventory_df['Reorder Point'], 'Health Status'] = 'Low'
        inventory_df.loc[inventory_df['Stock Levels'] <= 0, 'Health Status'] = 'Out of Stock'
        inventory_df.loc[inventory_df['Stock Levels'] > 0.8 * inventory_df['Warehouse Capacity'], 'Health Status'] = 'Overstocked'
        
        status_counts = inventory_df['Health Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Define colors for each status
        colors = {
            'Normal': '#4caf50',
            'Low': '#ff9800',
            'Out of Stock': '#f44336',
            'Overstocked': '#2196f3'
        }
        
        # Create treemap for inventory health
        fig = px.treemap(
            status_counts,
            path=['Status'],
            values='Count',
            color='Status',
            color_discrete_map=colors
        )
        
        fig.update_traces(textinfo="label+value+percent parent")
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Inventory by store/category
        col1, col2 = st.columns(2)
        
        with col1:
            # Inventory by store
            store_inventory = inventory_df.groupby('Store ID')['Stock Levels'].sum().reset_index()
            
            fig = px.bar(
                store_inventory,
                x='Store ID',
                y='Stock Levels',
                title="Inventory by Store",
                labels={'Stock Levels': 'Total Stock', 'Store ID': 'Store ID'},
                color='Stock Levels',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Inventory by category (assuming Category column exists)
            if 'Category' in inventory_df.columns:
                category_inventory = inventory_df.groupby('Category')['Stock Levels'].sum().reset_index()
                
                fig = px.pie(
                    category_inventory,
                    values='Stock Levels',
                    names='Category',
                    title="Inventory by Category",
                    hole=0.4
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create product type if category doesn't exist
                inventory_df['Product Type'] = inventory_df['Product ID'].apply(lambda x: f"Type {(x % 5) + 1}")
                type_inventory = inventory_df.groupby('Product Type')['Stock Levels'].sum().reset_index()
                
                fig = px.pie(
                    type_inventory,
                    values='Stock Levels',
                    names='Product Type',
                    title="Inventory by Product Type",
                    hole=0.4
                )
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    with tabs[1]:  # Stock Levels tab
        st.markdown("### Current Stock Levels")
        
        # Add search and filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_product = st.text_input("Search by Product ID")
        
        with col2:
            filter_status = st.multiselect(
                "Filter by Status",
                options=['Normal', 'Low', 'Out of Stock', 'Overstocked'],
                default=['Low', 'Out of Stock']
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=['Product ID', 'Stock Levels', 'Reorder Point', 'Days Until Reorder'],
                index=1
            )
        
        # Filter and sort data
        filtered_data = inventory_df.copy()
        
        if search_product:
            filtered_data = filtered_data[filtered_data['Product ID'].astype(str).str.contains(search_product)]
        
        if filter_status:
            filtered_data = filtered_data[filtered_data['Health Status'].isin(filter_status)]
        
        # Calculate days until reorder
        if 'Sales Rate (daily)' in filtered_data.columns:
            filtered_data['Days Until Reorder'] = (filtered_data['Stock Levels'] - filtered_data['Reorder Point']) / filtered_data['Sales Rate (daily)']
            filtered_data['Days Until Reorder'] = filtered_data['Days Until Reorder'].clip(lower=0)
        else:
            filtered_data['Days Until Reorder'] = 0
        
        # Sort data
        if sort_by == 'Days Until Reorder':
            filtered_data = filtered_data.sort_values(by='Days Until Reorder')
        else:
            filtered_data = filtered_data.sort_values(by=sort_by, ascending=(sort_by == 'Product ID'))
        
        # Display data as interactive table
        st.dataframe(
            filtered_data[['Product ID', 'Store ID', 'Stock Levels', 'Reorder Point', 'Health Status', 'Days Until Reorder']],
            height=400,
            use_container_width=True
        )
        
        # Stock level visualization
        st.markdown("### Stock Level Distribution")
        
        # Create histogram of stock levels
        fig = px.histogram(
            inventory_df,
            x='Stock Levels',
            nbins=20,
            marginal='box',
            color_discrete_sequence=['#1e88e5'],
            title="Distribution of Stock Levels"
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="Stock Level (units)",
            yaxis_title="Number of Products"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tabs[2]:  # Reorder Planning tab
        st.markdown("### Reorder Planning")
        
        # Identify items to reorder
        reorder_items = inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]
        
        # Display count of items to reorder
        st.info(f"🔍 {len(reorder_items)} products need to be reordered.")
        
        if not reorder_items.empty:
            # Create reorder plan with optimal order quantities
            reorder_plan = reorder_items.copy()
            
            # Calculate optimal order quantity
            reorder_plan['Order Quantity'] = (reorder_plan['Reorder Point'] * 2 - reorder_plan['Stock Levels']).astype(int)
            
            # Calculate total cost
            if 'Unit Cost' in reorder_plan.columns:
                reorder_plan['Total Cost'] = reorder_plan['Order Quantity'] * reorder_plan['Unit Cost']
            else:
                # Simulate unit cost if not available
                reorder_plan['Unit Cost'] = 10 + reorder_plan['Product ID'] % 10
                reorder_plan['Total Cost'] = reorder_plan['Order Quantity'] * reorder_plan['Unit Cost']
            
            # Calculate arrival date based on lead time
            if 'Supplier Lead Time (days)' in reorder_plan.columns:
                today = pd.Timestamp.now().date()
                reorder_plan['Expected Arrival'] = reorder_plan['Supplier Lead Time (days)'].apply(
                    lambda x: (pd.Timestamp.now() + pd.Timedelta(days=x)).strftime('%Y-%m-%d')
                )
            
            # Display reorder plan
            st.dataframe(
                reorder_plan[[
                    'Product ID', 'Store ID', 'Stock Levels', 'Reorder Point', 
                    'Order Quantity', 'Unit Cost', 'Total Cost', 'Expected Arrival'
                ]],
                height=300,
                use_container_width=True
            )
            
            # Summary of reorder plan
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_items = reorder_plan['Order Quantity'].sum()
                display_metric("Total Items to Order", f"{total_items:,.0f} units")
            
            with col2:
                total_cost = reorder_plan['Total Cost'].sum()
                display_metric("Total Cost", f"${total_cost:,.2f}")
            
            with col3:
                avg_lead_time = reorder_plan['Supplier Lead Time (days)'].mean() if 'Supplier Lead Time (days)' in reorder_plan.columns else 5
                display_metric("Avg. Lead Time", f"{avg_lead_time:.1f} days")
            
            # Prioritization of reorder items
            st.markdown("### Order Prioritization")
            
            # Calculate days of stock remaining
            if 'Sales Rate (daily)' in reorder_plan.columns:
                reorder_plan['Days of Stock'] = reorder_plan['Stock Levels'] / reorder_plan['Sales Rate (daily)']
                reorder_plan['Days of Stock'] = reorder_plan['Days of Stock'].fillna(0)
            else:
                # Simulate sales rate if not available
                reorder_plan['Sales Rate (daily)'] = 5 + (reorder_plan['Product ID'] % 5)
                reorder_plan['Days of Stock'] = reorder_plan['Stock Levels'] / reorder_plan['Sales Rate (daily)']
            
            # Create priority level
            reorder_plan['Priority'] = pd.cut(
                reorder_plan['Days of Stock'],
                bins=[-1, 1, 3, 7, float('inf')],
                labels=['Critical (0-1 days)', 'High (1-3 days)', 'Medium (3-7 days)', 'Low (7+ days)']
            )
            
            # Count by priority
            priority_counts = reorder_plan['Priority'].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Count']
            
            # Create priority chart
            fig = px.pie(
                priority_counts,
                values='Count',
                names='Priority',
                color='Priority',
                color_discrete_map={
                    'Critical (0-1 days)': '#f44336',
                    'High (1-3 days)': '#ff9800',
                    'Medium (3-7 days)': '#2196f3',
                    'Low (7+ days)': '#4caf50'
                },
                title="Order Priority Distribution"
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    with tabs[3]:  # Expiry Management tab
        st.markdown("### Expiry Management")
        
        # Check if expiry data is available
        if 'Expiry Date' in inventory_df.columns:
            # Calculate days until expiry
            today = pd.Timestamp.now().date()
            inventory_df['Days Until Expiry'] = (inventory_df['Expiry Date'] - pd.Timestamp(today)).dt.days
            
            # Identify near-expiry items
            expiry_window = st.slider("Expiry Window (days)", 1, 90, 30)
            near_expiry = inventory_df[inventory_df['Days Until Expiry'] <= expiry_window]
            
            st.info(f"🕒 {len(near_expiry)} products will expire in the next {expiry_window} days.")
            
            if not near_expiry.empty:
                # Create expiry risk categories
                near_expiry['Expiry Risk'] = pd.cut(
                    near_expiry['Days Until Expiry'],
                    bins=[-1, 7, 14, 30, float('inf')],
                    labels=['Critical (< 7 days)', 'High (7-14 days)', 'Medium (14-30 days)', 'Low (30+ days)']
                )
                
                # Display near-expiry items
                st.dataframe(
                    near_expiry[[
                        'Product ID', 'Store ID', 'Stock Levels', 
                        'Expiry Date', 'Days Until Expiry', 'Expiry Risk'
                    ]].sort_values('Days Until Expiry'),
                    height=300,
                    use_container_width=True
                )
                
                # Expiry risk visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution of days until expiry
                    fig = px.histogram(
                        near_expiry,
                        x='Days Until Expiry',
                        color='Expiry Risk',
                        labels={'Days Until Expiry': 'Days Until Expiry', 'count': 'Number of Products'},
                        title="Distribution of Expiry Dates",
                        color_discrete_map={
                            'Critical (< 7 days)': '#f44336',
                            'High (7-14 days)': '#ff9800',
                            'Medium (14-30 days)': '#2196f3',
                            'Low (30+ days)': '#4caf50'
                        }
                    )
                    
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                

# Pricing Optimization page
def display_pricing_optimization(merged_df):
    st.markdown("<h1 style='text-align: center;'>Pricing Optimization</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff;'>Optimize product pricing for maximum profit</p>", unsafe_allow_html=True)
    
    # Create tabs for different pricing views
    tabs = st.tabs(["Price Elasticity", "Competitive Analysis", "Price Recommendations", "What-If Analysis"])
    
    with tabs[0]:  # Price Elasticity tab
        st.markdown("### Price Elasticity Analysis")
        
        # Select product for analysis
        products = sorted(merged_df['Product ID'].unique())
        selected_product = st.selectbox("Select Product", products, key="elasticity_product")
        
        # Filter data for selected product
        product_data = merged_df[merged_df['Product ID'] == selected_product].copy()
        
        # Check if price and sales data are available
        if 'Price' in product_data.columns and 'Sales Volume' in product_data.columns:
            # Create synthetic price test data if real data is insufficient
            if len(product_data) < 5:
                base_price = product_data['Price'].mean()
                test_prices = [base_price * (1 + adj) for adj in [-0.2, -0.1, 0, 0.1, 0.2]]
                base_sales = product_data['Sales Volume'].mean()
                
                # Create elasticity model (price increases → sales decrease)
                elasticity = -1.3  # Elasticity coefficient
                test_sales = [base_sales * (price/base_price)**elasticity for price in test_prices]
                
                # Create synthetic data
                synthetic_data = pd.DataFrame({
                    'Price': test_prices,
                    'Sales Volume': test_sales,
                    'Revenue': [p * s for p, s in zip(test_prices, test_sales)]
                })
                
                plot_data = synthetic_data
                st.info("Using simulated price elasticity data based on limited historical data.")
            else:
                # Calculate revenue
                product_data['Revenue'] = product_data['Price'] * product_data['Sales Volume']
                plot_data = product_data
            
            # Create price elasticity scatter plot
            fig = px.scatter(
                plot_data,
                x='Price',
                y='Sales Volume',
                size='Revenue',
                labels={'Price': 'Price ($)', 'Sales Volume': 'Units Sold'},
                title=f"Price Elasticity for Product ID: {selected_product}",
                trendline="ols",
                size_max=20
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate optimal price
            # Simple quadratic model for revenue as function of price
            from scipy import optimize
            
            prices = plot_data['Price'].values
            revenues = plot_data['Revenue'].values
            
            def revenue_model(price, a, b, c):
                return a * price**2 + b * price + c
            
            # Fit model
            try:
                params, _ = optimize.curve_fit(revenue_model, prices, revenues, p0=[-1, 10, 0])
                a, b, c = params
                
                # Find price that maximizes revenue
                optimal_price = -b / (2 * a) if a < 0 else plot_data['Price'].mean()
                
                # Create price-revenue curve
                price_range = np.linspace(min(prices) * 0.8, max(prices) * 1.2, 100)
                predicted_revenue = revenue_model(price_range, a, b, c)
                
                # Create revenue optimization chart
                fig = go.Figure()
                
                # Add actual data points
                fig.add_trace(go.Scatter(
                    x=prices,
                    y=revenues,
                    mode='markers',
                    name='Actual Data',
                    marker=dict(color='#1e88e5', size=10)
                ))
                
                # Add predicted curve
                fig.add_trace(go.Scatter(
                    x=price_range,
                    y=predicted_revenue,
                    mode='lines',
                    name='Revenue Curve',
                    line=dict(color='#ff9800', width=2)
                ))
                
                # Add optimal price point
                max_revenue = revenue_model(optimal_price, a, b, c)
                fig.add_trace(go.Scatter(
                    x=[optimal_price],
                    y=[max_revenue],
                    mode='markers',
                    name='Optimal Price',
                    marker=dict(color='#4caf50', size=15, symbol='star')
                ))
                
                fig.update_layout(
                    title="Revenue Optimization Curve",
                    xaxis_title="Price ($)",
                    yaxis_title="Revenue ($)",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display optimal price and metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_price = plot_data['Price'].mean()
                    display_metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    display_metric("Optimal Price", f"${optimal_price:.2f}", 
                                  delta=f"{(optimal_price/current_price - 1)*100:.1f}%")
                
                with col3:
                    current_rev = plot_data['Revenue'].mean()
                    optimal_rev = revenue_model(optimal_price, a, b, c)
                    display_metric("Est. Revenue Increase", f"${optimal_rev - current_rev:.2f}", 
                                  delta=f"{(optimal_rev/current_rev - 1)*100:.1f}%")
            except:
                st.warning("Unable to calculate optimal price with current data. Try selecting a different product.")
        else:
            st.warning("Price and sales data are not available for elasticity analysis.")
    
    with tabs[1]:  # Competitive Analysis tab
        st.markdown("### Competitive Pricing Analysis")
        
        # Check if necessary columns exist
        if all(col in pricing_df.columns for col in ['Product ID', 'Price', 'Competitor Prices']):
            # Product selection
            selected_product = st.selectbox("Select Product", products, key="comp_analysis_product")
            
            # Filter for selected product
            product_data = pricing_df[pricing_df['Product ID'] == selected_product].copy()
            
            if not product_data.empty:
                # Calculate price difference
                product_data['Price Difference'] = product_data['Price'] - product_data['Competitor Prices']
                product_data['Price Difference %'] = (product_data['Price'] / product_data['Competitor Prices'] - 1) * 100
                
                # Calculate average prices
                avg_price = product_data['Price'].mean()
                avg_comp_price = product_data['Competitor Prices'].mean()
                avg_diff = avg_price - avg_comp_price
                avg_diff_pct = (avg_price / avg_comp_price - 1) * 100 if avg_comp_price > 0 else 0
                
                # Display price comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    display_metric("Your Price", f"${avg_price:.2f}")
                
                with col2:
                    display_metric("Competitor Prices", f"${avg_comp_price:.2f}")
                
                with col3:
                    display_metric("Price Difference", f"${avg_diff:.2f}", delta=f"{avg_diff_pct:.1f}%")
                
                # Create price comparison chart
                fig = go.Figure()
                
                # Add competitor price line
                fig.add_trace(go.Scatter(
                    x=product_data['Date'] if 'Date' in product_data.columns else list(range(len(product_data))),
                    y=product_data['Competitor Prices'],
                    mode='lines+markers',
                    name='Competitor Prices',
                    line=dict(color='#f44336', width=2)
                ))
                
                # Add your price line
                fig.add_trace(go.Scatter(
                    x=product_data['Date'] if 'Date' in product_data.columns else list(range(len(product_data))),
                    y=product_data['Price'],
                    mode='lines+markers',
                    name='Your Price',
                    line=dict(color='#2196f3', width=2)
                ))
                
                fig.update_layout(
                    title=f"Price Comparison for Product ID: {selected_product}",
                    xaxis_title="Time Period",
                    yaxis_title="Price ($)",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Price position in market
                st.markdown("### Price Position in Market")
                
                # Create market position visualization
                if 'Category' in product_data.columns:
                    category = product_data['Category'].iloc[0]
                    category_products = pricing_df[pricing_df['Category'] == category]
                    
                    # Calculate price range in category
                    min_price = category_products['Price'].min()
                    max_price = category_products['Price'].max()
                    avg_category_price = category_products['Price'].mean()
                    
                    # Create price position chart
                    fig = go.Figure()
                    
                    # Add price range bar
                    fig.add_trace(go.Bar(
                        x=[max_price - min_price],
                        y=['Price Range'],
                        orientation='h',
                        base=min_price,
                        marker_color='rgba(200, 200, 200, 0.5)',
                        hoverinfo='none'
                    ))
                    
                    # Add markers for key points
                    fig.add_trace(go.Scatter(
                        x=[min_price, avg_category_price, max_price, avg_price],
                        y=['Price Range', 'Price Range', 'Price Range', 'Price Range'],
                        mode='markers+text',
                        marker=dict(
                            color=['#f44336', '#ff9800', '#4caf50', '#2196f3'],
                            size=[12, 12, 12, 16],
                            symbol=['circle', 'circle', 'circle', 'star']
                        ),
                        text=['Min', 'Avg', 'Max', 'Your Price'],
                        textposition='top center'
                    ))
                    
                    fig.update_layout(
                        title=f"Price Position in {category} Category",
                        xaxis_title="Price ($)",
                        yaxis=dict(showticklabels=False),
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # If no category is available, use all products
                    all_prices = pricing_df['Price'].dropna()
                    min_price = all_prices.min()
                    max_price = all_prices.max()
                    avg_all_price = all_prices.mean()
                    
                    # Create price position chart
                    fig = go.Figure()
                    
                    # Add price range bar
                    fig.add_trace(go.Bar(
                        x=[max_price - min_price],
                        y=['Price Range'],
                        orientation='h',
                        base=min_price,
                        marker_color='rgba(200, 200, 200, 0.5)',
                        hoverinfo='none'
                    ))
                    
                    # Add markers for key points
                    fig.add_trace(go.Scatter(
                        x=[min_price, avg_all_price, max_price, avg_price],
                        y=['Price Range', 'Price Range', 'Price Range', 'Price Range'],
                        mode='markers+text',
                        marker=dict(
                            color=['#f44336', '#ff9800', '#4caf50', '#2196f3'],
                            size=[12, 12, 12, 16],
                            symbol=['circle', 'circle', 'circle', 'star']
                        ),
                        text=['Min', 'Avg', 'Max', 'Your Price'],
                        textposition='top center'
                    ))
                    
                    fig.update_layout(
                        title="Price Position Across All Products",
                        xaxis_title="Price ($)",
                        yaxis=dict(showticklabels=False),
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for selected product.")
        else:
            st.warning("Competitive price data is not available.")
    
    with tabs[2]:  # Price Recommendations tab
        st.markdown("### Price Optimization Recommendations")
        
        # Create recommendations based on elasticity and competitive data
        if all(col in merged_df.columns for col in ['Product ID', 'Price', 'Sales Volume', 'Competitor Prices']):
            # Calculate metrics for recommendations
            merged_df['Revenue'] = merged_df['Price'] * merged_df['Sales Volume']
            merged_df['Price Diff %'] = (merged_df['Price'] / merged_df['Competitor Prices'] - 1) * 100
            
            # Group by product
            product_metrics = merged_df.groupby('Product ID').agg({
                'Price': 'mean',
                'Competitor Prices': 'mean',
                'Sales Volume': 'mean',
                'Revenue': 'mean',
                'Price Diff %': 'mean'
            }).reset_index()
            
            # Calculate price elasticity (simplified)
            elasticity_data = {}
            for product in product_metrics['Product ID'].unique():
                product_data = merged_df[merged_df['Product ID'] == product]
                if len(product_data) >= 5:
                    try:
                        # Calculate log-log elasticity
                        X = np.log(product_data['Price'].values.reshape(-1, 1))
                        y = np.log(product_data['Sales Volume'].values)
                        
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression().fit(X, y)
                        elasticity_data[product] = model.coef_[0]
                    except:
                        elasticity_data[product] = -1.0  # Default elasticity
                else:
                    elasticity_data[product] = -1.0  # Default elasticity
            
            product_metrics['Elasticity'] = product_metrics['Product ID'].map(elasticity_data)
            
            # Generate price recommendations
            def recommend_price(row):
                elasticity = row['Elasticity']
                competitor_price = row['Competitor Prices']
                current_price = row['Price']
                
                if abs(elasticity) < 0.5:  # Inelastic
                    if current_price < competitor_price * 0.95:
                        return min(current_price * 1.05, competitor_price * 0.98)
                    else:
                        return current_price
                elif abs(elasticity) > 1.5:  # Highly elastic
                    if current_price > competitor_price * 1.05:
                        return max(current_price * 0.95, competitor_price * 1.02)
                    else:
                        return current_price
                else:  # Moderately elastic
                    if abs(current_price - competitor_price) / competitor_price > 0.1:
                        return (current_price + competitor_price) / 2
                    else:
                        return current_price
            
            product_metrics['Recommended Price'] = product_metrics.apply(recommend_price, axis=1)
            product_metrics['Price Change %'] = (product_metrics['Recommended Price'] / product_metrics['Price'] - 1) * 100
            
            # Display recommendations
            st.dataframe(
                product_metrics[[
                    'Product ID', 'Price', 'Competitor Prices', 'Price Diff %',
                    'Elasticity', 'Recommended Price', 'Price Change %'
                ]].sort_values(by='Price Change %', ascending=False),
                height=300,
                use_container_width=True
            )
            
            # Visualize price changes
            top_products = product_metrics.sort_values(by='Price Change %', key=abs, ascending=False).head(10)
            
            fig = go.Figure()
            
            # Add current price bars
            fig.add_trace(go.Bar(
                x=top_products['Product ID'].astype(str),
                y=top_products['Price'],
                name='Current Price',
                marker_color='#1e88e5'
            ))
            
            # Add recommended price bars
            fig.add_trace(go.Bar(
                x=top_products['Product ID'].astype(str),
                y=top_products['Recommended Price'],
                name='Recommended Price',
                marker_color='#4caf50'
            ))
            
            # Add competitor price points
            fig.add_trace(go.Scatter(
                x=top_products['Product ID'].astype(str),
                y=top_products['Competitor Prices'],
                mode='markers',
                name='Competitor Prices',
                marker=dict(color='#f44336', size=10, symbol='diamond')
            ))
            
            fig.update_layout(
                title="Top Price Change Recommendations",
                xaxis_title="Product ID",
                yaxis_title="Price ($)",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary of recommendations
            st.markdown("### Recommendation Summary")
            
            increases = product_metrics[product_metrics['Price Change %'] > 0]
            decreases = product_metrics[product_metrics['Price Change %'] < 0]
            unchanged = product_metrics[product_metrics['Price Change %'] == 0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_metric("Price Increases", f"{len(increases)} products", 
                              delta=f"{increases['Price Change %'].mean():.1f}% avg")
            
            with col2:
                display_metric("Price Decreases", f"{len(decreases)} products",
                              delta=f"{decreases['Price Change %'].mean():.1f}% avg")
            
            with col3:
                revenue_impact = ((product_metrics['Recommended Price'] * product_metrics['Sales Volume']).sum() - 
                                 (product_metrics['Price'] * product_metrics['Sales Volume']).sum())
                display_metric("Est. Revenue Impact", f"${revenue_impact:.2f}",
                              delta=f"{revenue_impact / (product_metrics['Price'] * product_metrics['Sales Volume']).sum() * 100:.1f}%")
        else:
            st.warning("Sufficient data for price recommendations is not available.")
    
    with tabs[3]:  # What-If Analysis tab
        st.markdown("### Price What-If Analysis")
        
        # Product selection
        selected_product = st.selectbox("Select Product", products, key="whatif_product")
        
        # Filter for selected product
        product_data = merged_df[merged_df['Product ID'] == selected_product].copy()
        
        if not product_data.empty and 'Price' in product_data.columns and 'Sales Volume' in product_data.columns:
            # Calculate current metrics
            current_price = product_data['Price'].mean()
            current_sales = product_data['Sales Volume'].mean()
            current_revenue = current_price * current_sales
            
            # Get elasticity (or estimate)
            try:
                X = np.log(product_data['Price'].values.reshape(-1, 1))
                y = np.log(product_data['Sales Volume'].values)
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                elasticity = model.coef_[0]
            except:
                elasticity = -1.3  # Default elasticity
            
            # What-if price slider
            st.markdown(f"#### Current Price: ${current_price:.2f}")
            price_pct_change = st.slider("Price Change (%)", -50, 50, 0)
            what_if_price = current_price * (1 + price_pct_change / 100)
            
            # Calculate projected metrics
            projected_sales = current_sales * (what_if_price / current_price) ** elasticity
            projected_revenue = what_if_price * projected_sales
            
            # Display projected impact
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_metric("Projected Price", f"${what_if_price:.2f}", 
                              delta=f"{price_pct_change:.1f}%")
            
            with col2:
                sales_change_pct = (projected_sales / current_sales - 1) * 100
                display_metric("Projected Sales", f"{projected_sales:.1f} units", 
                              delta=f"{sales_change_pct:.1f}%")
            
            with col3:
                revenue_change_pct = (projected_revenue / current_revenue - 1) * 100
                display_metric("Projected Revenue", f"${projected_revenue:.2f}", 
                              delta=f"{revenue_change_pct:.1f}%")
            
            # Create what-if simulation chart
            price_range_pct = np.linspace(-30, 30, 25)
            price_range = [current_price * (1 + p/100) for p in price_range_pct]
            
            # Calculate projected sales and revenue for each price point
            sales_projections = [current_sales * (p / current_price) ** elasticity for p in price_range]
            revenue_projections = [p * s for p, s in zip(price_range, sales_projections)]
            
            # Normalize to percentages for better visualization
            sales_pct_change = [(s / current_sales - 1) * 100 for s in sales_projections]
            revenue_pct_change = [(r / current_revenue - 1) * 100 for r in revenue_projections]
            
            # Create simulation chart
            fig = go.Figure()
            
            # Add sales line
            fig.add_trace(go.Scatter(
                x=price_range_pct,
                y=sales_pct_change,
                mode='lines',
                name='Sales Change %',
                line=dict(color='#1e88e5', width=2)
            ))
            
            # Add revenue line
            fig.add_trace(go.Scatter(
                x=price_range_pct,
                y=revenue_pct_change,
                mode='lines',
                name='Revenue Change %',
                line=dict(color='#4caf50', width=2)
            ))
            
            # Add current point
            fig.add_trace(go.Scatter(
                x=[price_pct_change],
                y=[revenue_change_pct],
                mode='markers',
                name='Selected Point',
                marker=dict(color='#ff9800', size=12, symbol='star')
            ))
            
            # Add zero line
            fig.add_shape(
                type="line",
                x0=-30, x1=30,
                y0=0, y1=0,
                line=dict(color="gray", width=1, dash="dash")
            )
            
            fig.update_layout(
                title="Impact of Price Changes on Sales and Revenue",
                xaxis_title="Price Change (%)",
                yaxis_title="Projected Change (%)",
                height=450,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Elasticity interpretation
            st.markdown("### Price Elasticity Interpretation")
            
            if abs(elasticity) < 0.5:
                advice = "Product demand is relatively inelastic. Price increases will likely increase total revenue."
                recommendation = "Consider gradual price increases to maximize revenue, monitoring for competitive effects."
            elif abs(elasticity) < 1.0:
                advice = "Product has low elasticity. Small price changes won't dramatically affect demand."
                recommendation = "There's room for moderate price optimization based on market position and goals."
            elif abs(elasticity) < 1.5:
                advice = "Product has moderate elasticity. Price changes will affect demand proportionally."
                recommendation = "Balance price with volume - consider competitive positioning and market share goals."
            else:
                advice = "Product is highly elastic. Price increases significantly reduce demand."
                recommendation = "Focus on competitive pricing and consider volume discounts to drive sales."
            
            st.info(f"**Elasticity: {elasticity:.2f}**. {advice}")
            st.success(f"**Recommendation:** {recommendation}")
        else:
            st.warning("Sufficient sales and price data is not available for what-if analysis.")

# Dashboard page
def display_dashboard(demand_df, inventory_df, pricing_df):
    st.markdown("<h1 style='text-align: center;'>Retail Inventory Optimization Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #ffffff;'>A comprehensive overview of your retail operation</p>", unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total sales calculation
        total_sales = (demand_df['Sales Quantity'] * demand_df['Price']).sum()
        display_metric(
            "Total Sales", 
            f"${total_sales:,.0f}",
            delta=5.2
        )
    
    with col2:
        # Inventory health
        low_stock_perc = len(inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]) / len(inventory_df) * 100
        display_metric(
            "Inventory Health", 
            f"{100 - low_stock_perc:.1f}%",
            delta=-2.3
        )
    
    with col3:
        # Average product margin
        avg_margin = ((pricing_df['Price'] - pricing_df['Storage Cost']) / pricing_df['Price']).mean() * 100
        display_metric(
            "Avg Product Margin", 
            f"{avg_margin:.1f}%",
            delta=1.8
        )
    
    with col4:
        # Customer satisfaction - from customer review ratings
        avg_rating = pricing_df['Customer Reviews'].mean()
        display_metric(
            "Customer Satisfaction", 
            f"{avg_rating:.1f}/5",
            delta=0.3
        )
    
    # Dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Sales Trend")
        
        # Prepare sales trend data
        sales_trend = demand_df.groupby('Date')[['Sales Quantity']].sum().reset_index()
        
        # Create the time series plot
        fig = px.line(
            sales_trend,
            x='Date',
            y='Sales Quantity',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="",
            yaxis_title="Units Sold"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Inventory Status")
        
        # Calculate inventory status counts
        inventory_status = {
            'Normal': len(inventory_df[(inventory_df['Stock Levels'] >= inventory_df['Reorder Point']) & 
                                      (inventory_df['Stock Levels'] <= 0.8 * inventory_df['Warehouse Capacity'])]),
            'Low': len(inventory_df[inventory_df['Stock Levels'] < inventory_df['Reorder Point']]),
            'Overstocked': len(inventory_df[inventory_df['Stock Levels'] > 0.8 * inventory_df['Warehouse Capacity']])
        }
        
        # Create pie chart
        fig = px.pie(
            values=list(inventory_status.values()),
            names=list(inventory_status.keys()),
            color=list(inventory_status.keys()),
            color_discrete_map={
                'Normal': '#4caf50',
                'Low': '#ff9800',
                'Overstocked': '#2196f3'
            },
            hole=0.4
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom row charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Selling Products")
        
        # Calculate top selling products
        top_products = demand_df.groupby('Product ID')['Sales Quantity'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_products.index,
            y=top_products.values,
            labels={'x': 'Product ID', 'y': 'Units Sold'},
            color=top_products.values,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Store Performance")
        
        # Calculate store performance
        store_sales = demand_df.groupby('Store ID')['Sales Quantity'].sum().reset_index()
        store_inventory = inventory_df.groupby('Store ID')['Stock Levels'].sum().reset_index()
        
        # Merge data
        store_perf = pd.merge(store_sales, store_inventory, on='Store ID')
        store_perf['Stock Turnover'] = store_perf['Sales Quantity'] / store_perf['Stock Levels']
        
        fig = px.scatter(
            store_perf,
            x='Stock Levels',
            y='Sales Quantity',
            size='Stock Turnover',
            color='Store ID',
            labels={
                'Stock Levels': 'Inventory Level',
                'Sales Quantity': 'Sales Volume'
            },
            hover_data=['Store ID', 'Stock Turnover']
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts and recommendations
    st.markdown("### Alerts & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Critical Alerts")
        
        # Find critical low stock items
        critical_items = inventory_df[inventory_df['Stock Levels'] < 0.5 * inventory_df['Reorder Point']]
        
        if not critical_items.empty:
            for _, item in critical_items.head(3).iterrows():
                st.markdown(f"""
                <div style="background-color: #ffebee; border-left: 4px solid #f44336; padding: 12px; margin-bottom: 10px; border-radius: 4px; color: black;">
                    <strong>⚠️ Critical Low Stock</strong><br>
                    Product ID {int(item['Product ID'])} at Store {int(item['Store ID'])}<br>
                    Current stock: {int(item['Stock Levels'])} units (Reorder point: {int(item['Reorder Point'])})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No critical alerts at this time.")
    
    with col2:
        st.markdown("#### AI Recommendations")
        
        st.markdown("""
        <div style="background-color: #e8f5e9; border-left: 4px solid #4caf50; padding: 12px; margin-bottom: 10px; border-radius: 4px; color: black;">
            <strong>💡 Price Optimization</strong><br>
            Consider 5-10% price increase for high-demand products with low elasticity index to improve margins.
        </div>
        
        <div style="background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 12px; margin-bottom: 10px; border-radius: 4px; color: black;">
            <strong>💡 Inventory Rebalancing</strong><br>
            Transfer excess inventory from Store 3 to Store 1 to optimize stock distribution.
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Run Forecast", key="dashboard_forecast"):
            st.info("Forecast initiated. Check Demand Forecasting page for results.")
    
    with col2:
        if st.button("📊 Price Analysis", key="dashboard_price"):
            st.info("Price analysis initiated. Check Pricing Optimization page for results.")
    
    with col3:
        if st.button("📦 Rebalance Stock", key="dashboard_rebalance"):
            st.info("Stock rebalancing plan generated. Check Inventory Management page for details.")
    
    with col4:
        if st.button("💬 Agent Conference", key="dashboard_agents"):
            st.info("Multi-agent conference initiated. Check Multi-Agent System page for discussion results.")

if __name__ == "__main__":
    main()