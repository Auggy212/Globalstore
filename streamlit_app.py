import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Global Superstore EDA", layout="wide")

# Title
st.title("Global Superstore: Exploratory Data Analysis & Deep-Learning Prep")

# 1. Load data
@st.cache_data
def load_data(path):
    return pd.read_excel(path, sheet_name='Orders')

file_path = 'Global Superstore.xlsx'  # Ensure this file is in the same directory
orders = load_data(file_path)

# Show raw data head
if st.sidebar.checkbox("Show Raw Data", False):
    st.subheader("Raw Orders Data")
    st.dataframe(orders.head())

# Sidebar selection for visuals
vis_options = [
    "Numeric Feature Distributions",
    "Correlation Heatmap",
    "Profit by Region",
    "Order Count by Category",
    "PCA of Numeric Features"
]
choice = st.sidebar.radio("Select Visualization", vis_options)

# Numeric columns
numeric_cols = ['Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost']
numeric_data = orders[numeric_cols].dropna()

# 2. Numeric Feature Distributions
if choice == "Numeric Feature Distributions":
    st.subheader("Distribution of Numeric Features")
    for col in numeric_cols:
        fig = px.histogram(numeric_data, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

# 3. Correlation Heatmap
elif choice == "Correlation Heatmap":
    st.subheader("Correlation Heatmap of Numeric Features")
    corr_matrix = numeric_data.corr()
    fig = px.imshow(corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# 4. Profit by Region
elif choice == "Profit by Region":
    st.subheader("Profit Distribution by Region")
    fig = px.box(orders, x='Region', y='Profit', title='Profit by Region')
    st.plotly_chart(fig, use_container_width=True)

# 5. Order Count by Category
elif choice == "Order Count by Category":
    st.subheader("Orders Count by Category")
    count_df = orders['Category'].value_counts().reset_index()
    count_df.columns = ['Category', 'Count']
    fig = px.bar(count_df, x='Category', y='Count', title='Category Counts')
    st.plotly_chart(fig, use_container_width=True)

# 6. PCA of Numeric Features
elif choice == "PCA of Numeric Features":
    st.subheader("PCA on Numeric Features, Colored by Category")
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_df['Category'] = orders['Category'].values

    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Category', title='PCA of Numeric Features')
    st.plotly_chart(fig, use_container_width=True)

    # Show explained variance
    explained = pca.explained_variance_ratio_
    st.markdown(f"**Explained Variance Ratio:** PC1: {explained[0]:.2f}, PC2: {explained[1]:.2f}")

# Footer or instructions
st.sidebar.markdown("---")
st.sidebar.info("Use this Streamlit app to explore the Global Superstore dataset. Select a visualization from the sidebar.")
