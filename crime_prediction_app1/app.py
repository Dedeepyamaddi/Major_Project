import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Crime Data Predictions",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load the trained model and preprocessing objects
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset
df = pd.read_csv("data_cts_violent_and_sexual_crime.csv")
df = df.dropna()

# Custom CSS for professional dashboard styling
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Main app styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Professional Navbar */
    .navbar {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        border-radius: 0 0 15px 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-decoration: none;
    }
    
    .navbar-nav {
        display: flex;
        list-style: none;
        margin: 0;
        padding: 0;
        gap: 2rem;
    }
    
    .nav-item {
        position: relative;
    }
    
    .nav-link {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-link:hover {
        background-color: rgba(255,255,255,0.1);
        transform: translateY(-2px);
    }
    
    .nav-link.active {
        background-color: #3498db;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    .navbar-user {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: white;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(45deg, #3498db, #2980b9);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #3498db;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .metric-change.positive {
        color: #27ae60;
    }
    
    .metric-change.negative {
        color: #e74c3c;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    .chart-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Crime prediction cards */
    .prediction-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 3px solid #3498db;
    }
    
    .prediction-title {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .prediction-probability {
        font-size: 1.1rem;
        font-weight: bold;
        color: #3498db;
    }
    
    .prediction-trend {
        font-size: 0.9rem;
        color: #27ae60;
        font-weight: 500;
    }
    
    /* Progress bars */
    .progress-bar {
        width: 100%;
        height: 8px;
        background-color: #ecf0f1;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #2980b9);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    .sidebar .sidebar-content .block-container {
        padding-top: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    /* Navbar button styling */
    div[data-testid="column"] .stButton > button[kind="secondary"] {
        background: white !important;
        color: #333 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        min-width: 120px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="column"] .stButton > button[kind="secondary"]:hover {
        background: #f8f9fa !important;
        color: #333 !important;
        border-color: #ccc !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
    }
    
    /* Active navbar button styling */
    div[data-testid="column"] .stButton > button[kind="primary"] {
        background: #ee5253 !important;
        color: white !important;
        border: 1px solid #ee5253 !important;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        min-width: 120px;
        box-shadow: 0 2px 5px rgba(238, 82, 83, 0.3);
    }
    
    div[data-testid="column"] .stButton > button[kind="primary"]:hover {
        background: #d64242 !important;
        color: white !important;
        border-color: #d64242 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 10px rgba(238, 82, 83, 0.4);
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Dashboard'

# Create Professional Navbar
def create_navbar():
    # Create navbar with logo using Streamlit columns - all in one row
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    with col1:
        try:
            st.image("Nec.png.jpg", width=50)
        except:
            st.markdown("🏛️")  # Fallback icon if image doesn't load
    
    with col2:
        st.markdown("""
        <div style="font-size: 1.5rem; font-weight: bold; color: black; margin-top: 0.5rem;">
        🔍Crime Data Predictions
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons - all in one row
    with col3:
        if st.button("📊 Dashboard", key="nav_dashboard", help="Go to Dashboard", 
                    type="primary" if st.session_state.current_page == 'Dashboard' else "secondary"):
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    
    with col4:
        if st.button("🔮 Predictions", key="nav_predictions", help="Go to Predictions", 
                    type="primary" if st.session_state.current_page == 'Predictions' else "secondary"):
            st.session_state.current_page = 'Predictions'
            st.rerun()
    
    with col5:
        if st.button("📈 Analysis", key="nav_analysis", help="Go to Analysis", 
                    type="primary" if st.session_state.current_page == 'Analysis' else "secondary"):
            st.session_state.current_page = 'Analysis'
            st.rerun()
    
    with col6:
        if st.button("📋 Reports", key="nav_reports", help="Go to Reports", 
                    type="primary" if st.session_state.current_page == 'Reports' else "secondary"):
            st.session_state.current_page = 'Reports'
            st.rerun()

# Create the navbar
create_navbar()

# Use session state for page navigation
page = st.session_state.current_page

# Main content based on selected page
if page == "Dashboard":
    # Key Metrics Row - Matching the exact layout from the image
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">363,528</div>
            <div class="metric-label">Total Crimes</div>
            <div class="metric-change positive">↗ 46</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">S23,8,618</div>
            <div class="metric-label">Prediction Accuracy</div>
            <div class="metric-change negative">↘ 1.6%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">733,338</div>
            <div class="metric-label">Violent Crimes</div>
            <div class="metric-change positive">↗ 216Z</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">306,418</div>
            <div class="metric-label">Sexual Crimes</div>
            <div class="metric-change positive">↗ 12.8%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Section - Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Crime Trends Over Time</div>', unsafe_allow_html=True)
        
        # Create sample time series data
        years = list(range(2015, 2024))
        violent_crimes = [320, 340, 360, 380, 400, 420, 440, 460, 480]
        sexual_crimes = [180, 190, 200, 210, 220, 230, 240, 250, 260]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=violent_crimes, name='Violent Crimes', line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=years, y=sexual_crimes, name='Sexual Crimes', line=dict(color='#e74c3c', width=3)))
        
        fig.update_layout(
            height=400,
            showlegend=True,
            xaxis_title="Year",
            yaxis_title="Number of Crimes",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🥧 Crime Distribution by Type</div>', unsafe_allow_html=True)
        
        # Create pie chart
        crime_types = ['Violent Crimes', 'Sexual Crimes', 'Property Crimes', 'Other']
        values = [45, 25, 20, 10]
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
        
        fig = go.Figure(data=[go.Pie(labels=crime_types, values=values, hole=0.3)])
        fig.update_traces(marker=dict(colors=colors))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section with crime predictions - Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🎯 Crime Predictions</div>', unsafe_allow_html=True)
        
        # Create sample prediction data matching the image
        predictions_data = {
            'Crime Type': ['Violent Assault', 'Sexual Violence', 'Property Theft', 'Domestic Violence', 'Robbery'],
            'Probability': [66.0, 61.3, 52.7, 68.2, 62.4],
            'Trend': ['+5.2%', '+1.9%', '+12%', '+34%', '+2.2%']
        }
        
        # Display predictions with custom styling
        for i in range(len(predictions_data['Crime Type'])):
            crime_type = predictions_data['Crime Type'][i]
            probability = predictions_data['Probability'][i]
            trend = predictions_data['Trend'][i]
            
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-title">{crime_type}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {probability}%"></div>
                </div>
                <div class="prediction-probability">{probability}%</div>
                <div class="prediction-trend">{trend}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Regional Crime Analysis</div>', unsafe_allow_html=True)
        
        # Create bar chart for regional analysis
        regions = ['North', 'South', 'East', 'West', 'Central']
        crime_counts = [120, 150, 180, 200, 160]
        
        fig = go.Figure(data=[
            go.Bar(x=regions, y=crime_counts, marker_color=['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6'])
        ])
        
        fig.update_layout(
            height=400,
            xaxis_title="Region",
            yaxis_title="Crime Count",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Predictions":
    st.markdown('<h1 class="main-header">Crime Prediction Interface</h1>', unsafe_allow_html=True)
    
    # Input Parameters Section - Moved to Top
    st.subheader('🔧 Input Parameters')
    
    # Create input fields in main content area
    col1, col2, col3 = st.columns(3)
    
    with col1:
        country = st.selectbox('Country', df['Country'].unique())
        region = st.selectbox('Region', df['Region'].unique())
        subregion = st.selectbox('Subregion', df['Subregion'].unique())
        dimension = st.selectbox('Dimension', df['Dimension'].unique())
    
    with col2:
        category = st.selectbox('Category', df['Category'].unique())
        sex = st.selectbox('Sex', df['Sex'].unique())
        age = st.selectbox('Age', df['Age'].unique())
        year = st.slider('Year', int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].mean()))
    
    with col3:
        unit_of_measurement = st.selectbox('Unit of measurement', df['Unit of measurement'].unique())
        value = st.number_input('VALUE', value=float(df['VALUE'].mean()))
        source = st.selectbox('Source', df['Source'].unique())
        iso3_code = st.selectbox('Iso3_code', df['Iso3_code'].unique())

    # Create input dataframe
    input_data = {
        'Iso3_code': iso3_code,
        'Country': country,
        'Region': region,
        'Subregion': subregion,
        'Dimension': dimension,
        'Category': category,
        'Sex': sex,
        'Age': age,
        'Year': year,
        'Unit of measurement': unit_of_measurement,
        'VALUE': value,
        'Source': source
    }
    input_df = pd.DataFrame(input_data, index=[0])

    # Display user input
    st.subheader('📋 Selected Parameters')
    st.dataframe(input_df, use_container_width=True)

    # Preprocess the user input
    input_df_processed = input_df.copy()

    for col in label_encoders:
        if col in input_df_processed.columns:
            le = label_encoders[col]
            # Handling unseen labels
            input_df_processed[col] = input_df_processed[col].apply(lambda x: x if x in le.classes_ else 'Other')
            le.classes_ = np.append(le.classes_, 'Other')
            input_df_processed[col] = le.transform(input_df_processed[col])

    numerical_cols = input_df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year') # Year might not need scaling

    if numerical_cols:
        input_df_processed[numerical_cols] = scaler.transform(input_df_processed[numerical_cols])

    # Make prediction button
    if st.button('🔮 Generate Prediction', type="primary"):
        prediction = model.predict(input_df_processed)
        predicted_indicator = label_encoders['Indicator'].inverse_transform(prediction)
        
        st.success(f'🎯 **Predicted Crime Indicator:** {predicted_indicator[0]}')
        
        # Display prediction confidence
        prediction_proba = model.predict_proba(input_df_processed)
        max_confidence = np.max(prediction_proba) * 100
        
        st.info(f'📊 **Prediction Confidence:** {max_confidence:.1f}%')
        
        # Show probability distribution
        st.subheader("📊 Prediction Probability Distribution")
        prob_df = pd.DataFrame({
            'Class': [label_encoders['Indicator'].inverse_transform([i])[0] for i in range(len(prediction_proba[0]))],
            'Probability': prediction_proba[0] * 100
        })
        
        fig = px.bar(prob_df, x='Class', y='Probability', 
                    title="Probability Distribution for Each Class",
                    color='Probability',
                    color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model Performance Section - Moved to Bottom
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")
    
    # Load model metrics
    try:
        with open('model_metrics.pkl', 'rb') as f:
            model_metrics = pickle.load(f)
        model_accuracy = model_metrics['accuracy']
        classification_report = model_metrics['classification_report']
        confusion_matrix = np.array(model_metrics['confusion_matrix'])
        
        # Accuracy metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{model_accuracy*100:.1f}%</div>
                <div class="metric-label">Overall Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            macro_avg = classification_report['macro avg']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{macro_avg['precision']:.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{macro_avg['recall']:.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{macro_avg['f1-score']:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts Section
        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("🎯 Confusion Matrix")
            fig = px.imshow(confusion_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Confusion Matrix",
                           labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.subheader("📈 Classification Performance")
            classes = list(classification_report.keys())[:-3]
            metrics_data = []
            
            for class_name in classes:
                if isinstance(class_name, str) and class_name not in ['macro avg', 'weighted avg', 'accuracy']:
                    metrics_data.append({
                        'Class': f'Class {class_name}',
                        'Precision': classification_report[class_name]['precision'],
                        'Recall': classification_report[class_name]['recall'],
                        'F1-Score': classification_report[class_name]['f1-score']
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Precision', x=metrics_df['Class'], y=metrics_df['Precision'], marker_color='#1f77b4'))
                fig.add_trace(go.Bar(name='Recall', x=metrics_df['Class'], y=metrics_df['Recall'], marker_color='#ff7f0e'))
                fig.add_trace(go.Bar(name='F1-Score', x=metrics_df['Class'], y=metrics_df['F1-Score'], marker_color='#2ca02c'))
                
                fig.update_layout(
                    height=400,
                    title="Performance Metrics by Class",
                    xaxis_title="Class",
                    yaxis_title="Score",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
    except FileNotFoundError:
        st.error("Model metrics not found. Please retrain the model first.")

elif page == "Analysis":
    st.markdown('<h1 class="main-header">Crime Analysis</h1>', unsafe_allow_html=True)
    
    # Analysis content
    st.subheader("📈 Advanced Crime Analysis")
    
    # Sample analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crime Patterns by Age Group")
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        crime_rates = [45, 35, 25, 20, 15]
        
        fig = px.bar(x=age_groups, y=crime_rates, title="Crime Rate by Age Group")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Geographic Crime Distribution")
        # Create a sample heatmap
        st.info("Geographic heatmap would be displayed here")
    
    # Additional analysis content
    st.markdown("---")
    st.subheader("🔍 Crime Trend Analysis")
    
    # Time series analysis
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    violent_trend = [120, 135, 150, 140, 160, 170, 180, 175, 190, 185, 200, 195]
    sexual_trend = [80, 85, 90, 95, 100, 105, 110, 108, 115, 112, 120, 118]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=violent_trend, name='Violent Crimes', line=dict(color='#e74c3c', width=3)))
    fig.add_trace(go.Scatter(x=months, y=sexual_trend, name='Sexual Crimes', line=dict(color='#3498db', width=3)))
    
    fig.update_layout(
        height=400,
        title="Monthly Crime Trends",
        xaxis_title="Month",
        yaxis_title="Number of Crimes",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Reports":
    st.markdown('<h1 class="main-header">Crime Reports</h1>', unsafe_allow_html=True)
    
    # Reports content
    st.subheader("📊 Crime Reports and Statistics")
    
    # Sample report data
    report_data = {
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'Violent Crimes': [120, 135, 150, 140, 160, 170],
        'Sexual Crimes': [80, 85, 90, 95, 100, 105],
        'Property Crimes': [200, 210, 220, 230, 240, 250]
    }
    
    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True)
    
    # Additional report content
    st.markdown("---")
    st.subheader("📈 Monthly Crime Summary")
    
    # Create summary charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crime Distribution")
        crime_types = ['Violent', 'Sexual', 'Property', 'Other']
        values = [35, 25, 30, 10]
        
        fig = px.pie(values=values, names=crime_types, title="Crime Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regional Performance")
        regions = ['North', 'South', 'East', 'West']
        performance = [85, 92, 78, 88]
        
        fig = px.bar(x=regions, y=performance, title="Regional Crime Prevention Performance")
        st.plotly_chart(fig, use_container_width=True)