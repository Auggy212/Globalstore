import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(numeric_data[col], bins=30, edgecolor='black')
        axes[i].set_title(f"{col}")
    # Hide any extra subplot (axes[5] unused)
    if len(numeric_cols) < len(axes):
        fig.delaxes(axes[-1])
    plt.tight_layout()
    st.pyplot(fig)

# 3. Correlation Heatmap
elif choice == "Correlation Heatmap":
    st.subheader("Correlation Heatmap of Numeric Features")
    corr_matrix = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

# 4. Profit by Region
elif choice == "Profit by Region":
    st.subheader("Profit Distribution by Region")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Region', y='Profit', data=orders, ax=ax)
    ax.set_title('Profit by Region')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# 5. Order Count by Category
elif choice == "Order Count by Category":
    st.subheader("Orders Count by Category")
    count_series = orders['Category'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    count_series.plot(kind='bar', edgecolor='black', ax=ax)
    ax.set_ylabel('Count')
    ax.set_title('Category Counts')
    st.pyplot(fig)

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

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Category', data=pca_df, alpha=0.6, ax=ax)
    ax.set_title('PCA of Numeric Features')
    st.pyplot(fig)
    
    # Show explained variance
    explained = pca.explained_variance_ratio_
    st.markdown(f"**Explained Variance Ratio:** PC1: {explained[0]:.2f}, PC2: {explained[1]:.2f}")

# Footer or instructions
st.sidebar.markdown("---")
st.sidebar.info("Use this Streamlit app to explore the Global Superstore dataset. Select a visualization from the sidebar.")
