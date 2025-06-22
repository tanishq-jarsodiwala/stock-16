import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import base64
from sklearn.preprocessing import StandardScaler
import requests  # for Hugging Face API


# Enhanced CSS with animations and styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main background */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), 
                    url('https://i.postimg.cc/TY5DvhZq/pexels-peng-liu-45946-169647.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb, .css-17eq0hr, .css-6qob1r {
        background: linear-gradient(rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.9)) !important;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Keyframe animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(0, 123, 255, 0.8);
        }
    }
    
    /* Title styling */
    .main h1 {
        color: #1e3a8a;
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 1s ease-out;
    }
    
    /* Subtitle and descriptions */
    .main p {
        color: #374151;
        font-size: 1.1rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Enhanced marquee */
    .marquee {
        white-space: nowrap;
        overflow: hidden;
        position: relative;
        background: #45b7d1;
        color: white;
        padding: 12px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .marquee span {
        display: inline-block;
        position: absolute;
        will-change: transform;
        animation: marqueeMove 10s linear infinite;
    }

    @keyframes marqueeMove {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }

    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 5px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        color: #1e3a8a;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 58, 138, 0.1);
        transform: translateY(-2px);
        animation: glow 2s infinite;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #4f46e5;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        animation: pulse 1s infinite;
    }
    
    /* Form submit button in Tab 3 */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.3s ease !important;
        color: #374151 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stSelectbox select {
        color: #374151 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stNumberInput > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stNumberInput input {
        color: #374151 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stTextInput input {
        color: #374151 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        color: #374151 !important;
    }
    
    .stMultiSelect input {
        color: #374151 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Text area styling for Tab 3 chat */
    .stTextArea > div > div {
        background: rgba(30, 30, 30, 0.95) !important;
        border-radius: 12px !important;
        border: 2px solid #444 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea > div > div:focus-within {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
    }
    
    .stTextArea textarea {
        color: white !important;
        background: rgba(30, 30, 30, 0.95) !important;
        font-size: 16px !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1cypcdb h1, .css-1cypcdb h2, .css-1cypcdb h3,
    .css-17eq0hr h1, .css-17eq0hr h2, .css-17eq0hr h3,
    .css-6qob1r h1, .css-6qob1r h2, .css-6qob1r h3 {
        color: #ffffff !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5) !important;
    }
    
    .css-1d391kg p, .css-1d391kg div, .css-1d391kg span, .css-1d391kg label,
    .css-1cypcdb p, .css-1cypcdb div, .css-1cypcdb span, .css-1cypcdb label,
    .css-17eq0hr p, .css-17eq0hr div, .css-17eq0hr span, .css-17eq0hr label,
    .css-6qob1r p, .css-6qob1r div, .css-6qob1r span, .css-6qob1r label {
        color: #f3f4f6 !important;
    }
    
    /* Sidebar input styling */
    .css-1d391kg .stTextInput > div > div,
    .css-1cypcdb .stTextInput > div > div,
    .css-17eq0hr .stTextInput > div > div,
    .css-6qob1r .stTextInput > div > div {
        background: rgba(30, 30, 30, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #444 !important;
    }
    
    .css-1d391kg .stTextInput input,
    .css-1cypcdb .stTextInput input,
    .css-17eq0hr .stTextInput input,
    .css-6qob1r .stTextInput input {
        color: white !important;
        background: transparent !important;
    }
    
    .css-1d391kg .stSelectbox > div > div,
    .css-1cypcdb .stSelectbox > div > div,
    .css-17eq0hr .stSelectbox > div > div,
    .css-6qob1r .stSelectbox > div > div {
        background: rgba(30, 30, 30, 0.9) !important;
        border-radius: 12px !important;
        border: 2px solid #444 !important;
    }
    
    .css-1d391kg .stSelectbox select,
    .css-1cypcdb .stSelectbox select,
    .css-17eq0hr .stSelectbox select,
    .css-6qob1r .stSelectbox select {
        color: white !important;
        background: rgba(30, 30, 30, 0.9) !important;
    }
    
    /* Chat bubble styling */
    .chat-bubble {
        background-color: rgba(30, 30, 30, 0.95) !important;
        color: white !important;
        padding: 12px 16px !important;
        margin: 8px 0 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    .chat-user {
        border-left: 5px solid #5b9bd5 !important;
    }
    .chat-assistant {
        border-left: 5px solid #66bb6a !important;
    }
    
    /* Card-like containers */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Success and error messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        animation: slideInLeft 0.6s ease-out;
    }
    
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border-radius: 12px;
        animation: slideInLeft 0.6s ease-out;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(79, 70, 229, 0.1) !important;
        transform: translateX(5px) !important;
    }
    
    /* Sidebar expander styling */
    .css-1d391kg .streamlit-expanderHeader,
    .css-1cypcdb .streamlit-expanderHeader,
    .css-17eq0hr .streamlit-expanderHeader,
    .css-6qob1r .streamlit-expanderHeader {
        background: rgba(30, 30, 30, 0.9) !important;
        color: white !important;
        border: 1px solid #444 !important;
    }
    
    .css-1d391kg .streamlit-expanderContent,
    .css-1cypcdb .streamlit-expanderContent,
    .css-17eq0hr .streamlit-expanderContent,
    .css-6qob1r .streamlit-expanderContent {
        background: rgba(30, 30, 30, 0.9) !important;
        color: white !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Loading animation */
    .stSpinner {
        animation: pulse 1.5s infinite;
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Download button special styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        transform: translateY(-2px);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            margin: 0.5rem;
            padding: 1rem;
        }
        
        .main h1 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler used during model training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load company data
company_data = pd.read_csv("Equity.csv")
company_names = company_data['Company_Name'].tolist()

# Load FAQ data
faq_data = pd.read_csv("faq_stock_market_platform.csv")

# Function to adjust features based on the correlation matrix
def adjust_features(features, correlation_matrix):
    adjusted_features = features.copy()
    for feature_name in features.columns:
        adjustment = sum(
            features[other_feature] * correlation_matrix[feature_name].get(other_feature, 0)
            for other_feature in features.columns
        )
        adjusted_features[feature_name] = features[feature_name] + adjustment
    return adjusted_features

# App title and disclaimer
st.title("📈 NextGen Investors")
st.write("💼 Get stock recommendations based on fundamental analysis.")
st.markdown(
    """
    <marquee class="marquee" behavior="scroll" direction="left">
        ⚠ Disclaimer: This platform offers stock guidance based on fundamental analysis and machine learning but does not constitute financial advice; invest responsibly.
    </marquee>
    """, unsafe_allow_html=True
)

with st.sidebar:
    st.header("📖 FAQ Section")
    search_query = st.text_input("🔍 Search FAQs", help="Type a keyword to search for relevant questions")

    if search_query:
        filtered_faq_data = faq_data[faq_data['Question'].str.contains(search_query, case=False, na=False)]
    else:
        filtered_faq_data = faq_data

    if not filtered_faq_data.empty:
        selected_question = st.selectbox("Choose a Question", filtered_faq_data['Question'].tolist())
        answer = filtered_faq_data[filtered_faq_data['Question'] == selected_question]['Answer'].values[0]
        st.write("Answer:", answer)
    else:
        st.write("No questions match your search. Try a different keyword.")

    # Metrics Guide
    st.header("📊 Metrics Guide")
    with st.expander("📘 View Metrics Descriptions"):
        st.write("""
        - P/E Ratio: Indicates stock price compared to earnings.
        - P/B Ratio: Shows valuation based on book value.
        - D/E Ratio: Reflects the company’s debt compared to its equity.
        - ROE: Measures company profitability.
        - ROA: Shows how well assets are utilized.
        """)

    st.write("🔗 [Investopedia Guide](https://www.investopedia.com/terms/f/financialratios.asp)")

correlation_matrix = {
    "P/E Ratio": {"P/E Ratio": 1.0, "P/B Ratio": 0.0045, "D/E Ratio": -0.0035, "ROE": -0.0178, "ROA": -0.0105},
    "P/B Ratio": {"P/E Ratio": 0.0045, "P/B Ratio": 1.0, "D/E Ratio": 0.8175, "ROE": 0.0281, "ROA": 0.0153},
    "D/E Ratio": {"P/E Ratio": -0.0035, "P/B Ratio": 0.8175, "D/E Ratio": 1.0, "ROE": 0.0678, "ROA": -0.0069},
    "ROE": {"P/E Ratio": -0.0178, "P/B Ratio": 0.0281, "D/E Ratio": 0.0678, "ROE": 1.0, "ROA": 0.0771},
    "ROA": {"P/E Ratio": -0.0105, "P/B Ratio": 0.0153, "D/E Ratio": -0.0069, "ROE": 0.0771, "ROA": 1.0},
}

# Load stock data
@st.cache_data
def load_stock_data():
    df = pd.read_csv("Redefined_Cleaned_Stock_Data.csv")
    df.rename(columns={"Debt-to-Equity Ratio": "D/E Ratio"}, inplace=True)
    df["Company_Name_clean"] = df["Company_Name"].str.lower().str.strip()
    return df

stock_ratios = load_stock_data()
company_names = sorted(stock_ratios["Company_Name"].dropna().unique())

tab1, tab2, tab3 = st.tabs([
    "📊 Compare Companies",
    "📈 Individual Recommendation",
    "💬 Chat with AI"
])


with tab1:
    st.header("🔍 Compare Multiple Companies")
    selected_companies = st.multiselect(
        "Choose Companies to Compare", company_names,
        help="Select multiple companies to compare."

    )

    if selected_companies:
        st.write("### 📊 Enter or Edit Financial Metrics for Selected Companies")
        metrics_data = {}
        for company in selected_companies:
            st.subheader(company)
            with st.expander(f"📉 Financial Metrics for {company}"):
                clean_name = company.lower().strip()
                default_row = stock_ratios[stock_ratios['Company_Name_clean'] == clean_name]
                if not default_row.empty:
                    row = default_row.iloc[0]
                    pe_default = row.get("P/E Ratio", 0.0)
                    pb_default = row.get("P/B Ratio", 0.0)
                    de_default = row.get("D/E Ratio", 0.0)
                    roe_default = row.get("ROE", 0.0)
                    roa_default = row.get("ROA", 0.0)
                else:
                    pe_default = pb_default = de_default = roe_default = roa_default = 0.0
                metrics_data[company] = {
                    "P/E Ratio": st.number_input(f"{company} - P/E Ratio", format="%.2f", value=pe_default),
                    "P/B Ratio": st.number_input(f"{company} - P/B Ratio", format="%.2f", value=pb_default),
                    "D/E Ratio": st.number_input(f"{company} - D/E Ratio", format="%.2f", value=de_default),
                    "ROE": st.number_input(f"{company} - ROE", format="%.2f", value=roe_default),
                    "ROA": st.number_input(f"{company} - ROA", format="%.2f", value=roa_default)
                }

        if st.button("💡 Get Recommendations for Selected Companies"):
            comparison_results = []
            for company, metrics in metrics_data.items():
                features = pd.DataFrame([metrics])
                adjusted_features = adjust_features(features, correlation_matrix)
                scaled = scaler.transform(adjusted_features)
                prediction = model.predict(scaled)
                prediction_proba = model.predict_proba(scaled)
                recommendation = "Buy" if prediction[0] == 1 else "Do not buy"
                confidence = prediction_proba[0][1] * 100 if prediction[0] == 1 else prediction_proba[0][0] * 100
                comparison_results.append((company, recommendation, confidence, metrics))

            result_data = [{
                "Company": r[0],
                "Recommendation": r[1],
                "Confidence (%)": r[2],
                **r[3]
            } for r in comparison_results]

            comparison_df = pd.DataFrame(result_data)
            comparison_df.sort_values(by="Confidence (%)", ascending=True, inplace=True)
            st.write("### Sorted Comparison Table (by Confidence)")
            st.dataframe(comparison_df)

            chart_data = comparison_df[["Company", "P/E Ratio", "P/B Ratio", "D/E Ratio", "ROE", "ROA"]].melt(
                id_vars="Company", var_name="Metric", value_name="Value"
            )
            fig = px.bar(chart_data, x="Company", y="Value", color="Metric", barmode="group",
                         title="Comparison of Financial Metrics")
            st.plotly_chart(fig)
            csv_comp = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv_comp, "comparison_results.csv", "text/csv")

with tab2:
    st.header("📈 Individual Stock Recommendation")
    selected_company = st.selectbox("Choose a Company", company_names)
    st.write(f"Selected Company: {selected_company}")

    clean_name = selected_company.lower().strip()
    default_values = stock_ratios[stock_ratios['Company_Name_clean'] == clean_name]

    if not default_values.empty:
        row = default_values.iloc[0]
        p_e_default = row.get("P/E Ratio", 0.0)
        p_b_default = row.get("P/B Ratio", 0.0)
        d_e_default = row.get("D/E Ratio", 0.0)
        roe_default = row.get("ROE", 0.0)
        roa_default = row.get("ROA", 0.0)
    else:
        p_e_default = p_b_default = d_e_default = roe_default = roa_default = 0.0

    p_e_ratio = st.number_input("P/E Ratio", format="%.2f", value=p_e_default)
    p_b_ratio = st.number_input("P/B Ratio", format="%.2f", value=p_b_default)
    d_e_ratio = st.number_input("D/E Ratio", format="%.2f", value=d_e_default)
    roe = st.number_input("Return on Equity (ROE)", format="%.2f", value=roe_default)
    roa = st.number_input("Return on Assets (ROA)", format="%.2f", value=roa_default)

    if st.button("💡 Get Recommendation for Selected Company"):
        features = pd.DataFrame([{
            "P/E Ratio": p_e_ratio,
            "P/B Ratio": p_b_ratio,
            "D/E Ratio": d_e_ratio,
            "ROE": roe,
            "ROA": roa
        }])

        adjusted_features = adjust_features(features, correlation_matrix)
        scaled = scaler.transform(adjusted_features)
        prediction = model.predict(scaled)
        prediction_proba = model.predict_proba(scaled)

        recommendation = "✅ Buy the stock!" if prediction[0] == 1 else "❌ Do not buy the stock."
        confidence = prediction_proba[0][1] * 100 if prediction[0] == 1 else prediction_proba[0][0] * 100
        color = "green" if prediction[0] == 1 else "red"
        st.markdown(
            f"<span style='color:{color}; font-size:20px'>{recommendation} with {confidence:.2f}% confidence</span>",
            unsafe_allow_html=True
        )

        metrics_df = pd.DataFrame({
            'Metrics': ['P/E Ratio', 'P/B Ratio', 'D/E Ratio', 'ROE', 'ROA'],
            'Values': [p_e_ratio, p_b_ratio, d_e_ratio, roe, roa]
        })
        fig = px.bar(metrics_df, x='Metrics', y='Values', title=f"{selected_company} Financial Metrics")
        st.plotly_chart(fig)

        individual_df = pd.DataFrame([{
            "Company": selected_company,
            "Recommendation": recommendation,
            "Confidence (%)": confidence,
            "P/E Ratio": p_e_ratio,
            "P/B Ratio": p_b_ratio,
            "D/E Ratio": d_e_ratio,
            "ROE": roe,
            "ROA": roa
        }])
        csv_individual = individual_df.to_csv(index=False).encode('utf-8')
   
        st.download_button("📥 Download Recommendation", csv_individual, "individual_recommendation.csv", "text/csv")


with tab3:
    st.header("💬 Stock & Finance Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
elif st.session_state.chat_history and isinstance(st.session_state.chat_history[0], tuple):
    # Fix old format if needed
    st.session_state.chat_history = [
        {"role": "user" if sender.lower() == "user" else "assistant", "content": msg}
        for sender, msg in st.session_state.chat_history
    ]

# Apply custom styling
st.markdown("""
<style>
.chat-bubble {
    background-color: #111;
    color: white;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 12px;
    font-size: 16px;
    line-height: 1.5;
}
.chat-user {
    border-left: 5px solid #5b9bd5;
}
.chat-assistant {
    border-left: 5px solid #66bb6a;
}
textarea {
    background-color: #000 !important;
    color: white !important;
    font-size: 16px !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# Show past conversation (in flow)
for msg in st.session_state.chat_history:
    sender = msg["role"]
    content = msg["content"]
    role_class = "chat-user" if sender == "user" else "chat-assistant"
    sender_name = "🧑 User" if sender == "user" else "🤖 Assistant"
    st.markdown(f"<div class='chat-bubble {role_class}'><strong>{sender_name}:</strong> {content}</div>", unsafe_allow_html=True)

# Input section
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("You:", key="chat_input", label_visibility="collapsed", placeholder="Type your message...", height=60)
    send_button = st.form_submit_button("Send")

# On submit
if send_button and user_input.strip():
    # Temporarily store user message separately for sending
    new_message = {"role": "user", "content": user_input}

    with st.spinner("🤖 Assistant is typing..."):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }

        # Limit previous history for speed
        limited_history = st.session_state.chat_history[-6:] + [new_message]

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant with stock and finance knowledge."},
                *limited_history
            ],
            "temperature": 0.7
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            assistant_reply = response.json()["choices"][0]["message"]["content"].strip()

            # Append user and assistant messages to full history
            st.session_state.chat_history.append(new_message)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            st.error(f"❌ API Error: {e}")


