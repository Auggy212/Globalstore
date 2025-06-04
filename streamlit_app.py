import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Global Superstore Deep Dive", layout="wide")

st.title("📊 Global Superstore Deep Learning Exploration")

# File upload
uploaded_file = st.file_uploader("Upload the Global Superstore Excel file", type=["xlsx"])

@st.cache_data
def load_data(file):
    xls = pd.ExcelFile(file)
    df = xls.parse("Orders")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Sidebar filters
    region = st.sidebar.multiselect("Select Region(s)", options=df["Region"].unique(), default=list(df["Region"].unique()))
    category = st.sidebar.multiselect("Select Category(ies)", options=df["Category"].unique(), default=list(df["Category"].unique()))

    df_filtered = df[(df["Region"].isin(region)) & (df["Category"].isin(category))]

    # Section A: Visualizations Without Feature Engineering
    st.header("📌 Visualizations (Without Feature Engineering)")

    st.subheader("1. Monthly Sales and Profit Trend")
    monthly = df_filtered.resample('M', on='Order Date')[['Sales', 'Profit']].sum()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    monthly.plot(ax=ax1)
    ax1.set_title("Monthly Sales and Profit")
    ax1.set_ylabel("USD")
    ax1.set_xlabel("Month")
    st.pyplot(fig1)

    st.subheader("2. Sales by Category and Sub-Category")
    cat_sales = df_filtered.groupby(['Category', 'Sub-Category'])['Sales'].sum().sort_values()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    cat_sales.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_title("Sales by Category/Sub-Category")
    ax2.set_xlabel("Sales (USD)")
    st.pyplot(fig2)

    st.subheader("3. Discount vs Profit Scatterplot")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df_filtered, x="Discount", y="Profit", hue="Category", ax=ax3)
    ax3.set_title("Discount vs Profit")
    st.pyplot(fig3)

    st.subheader("4. Order Priority vs Shipping Mode")
    heatmap_data = pd.crosstab(df_filtered['Order Priority'], df_filtered['Ship Mode'])
    fig4, ax4 = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="Blues", ax=ax4)
    ax4.set_title("Order Priority vs Shipping Mode")
    st.pyplot(fig4)

    st.divider()

    # Section B: Feature Engineering
    st.header("🧠 Visualizations (With Feature Engineering)")

    df_filtered['Shipping Duration'] = (df_filtered['Ship Date'] - df_filtered['Order Date']).dt.days
    df_filtered['Profit Margin'] = df_filtered['Profit'] / df_filtered['Sales']
    df_filtered = df_filtered.replace([float("inf"), float("-inf")], None).dropna()

    st.subheader("5. Average Shipping Duration by Region")
    ship_time = df_filtered.groupby("Region")["Shipping Duration"].mean().sort_values()
    fig5, ax5 = plt.subplots()
    ship_time.plot(kind='barh', color='orange', ax=ax5)
    ax5.set_title("Avg Shipping Duration by Region")
    ax5.set_xlabel("Days")
    st.pyplot(fig5)

    st.subheader("6. Profit Margin Distribution")
    fig6, ax6 = plt.subplots()
    sns.histplot(df_filtered["Profit Margin"], kde=True, bins=30, ax=ax6)
    ax6.set_title("Distribution of Profit Margin")
    st.pyplot(fig6)

    st.subheader("7. Sales vs Profit Margin")
    fig7, ax7 = plt.subplots()
    sns.scatterplot(data=df_filtered, x="Sales", y="Profit Margin", hue="Region", ax=ax7)
    ax7.set_title("Sales vs Profit Margin")
    st.pyplot(fig7)

    st.success("✅ Analysis Complete. Use sidebar to filter data and explore different insights.")
else:
    st.info("Please upload the Global Superstore dataset to begin.")
