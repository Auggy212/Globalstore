import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras
import kerastuner as kt

# ---------------------------------------
# 1. DATA LOADING & CACHING
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel("Global Superstore.xlsx", sheet_name="Orders")
    # Basic cleanup: drop rows with missing critical fields
    df = df.dropna(subset=["Order Date", "Ship Date", "Sales", "Profit", "Quantity", "Discount"])
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df

# ---------------------------------------
# 2. SIDEBAR FILTERS (APPLY TO ALL SECTIONS)
# ---------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Global Superstore Deep Learning Explorer",
    initial_sidebar_state="expanded"
)

df = load_data()
st.sidebar.markdown("## 🔎 Data Filters")

# Date Range Filter
min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
date_range = st.sidebar.date_input(
    "Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["Order Date"].dt.date >= start_date) & (df["Order Date"].dt.date <= end_date)
filtered_df = df.loc[mask].copy()

# Category Filter
all_categories = sorted(filtered_df["Category"].unique())
selected_cats = st.sidebar.multiselect(
    "Categories",
    options=all_categories,
    default=all_categories
)
if selected_cats:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_cats)]

# Region Filter
all_regions = sorted(filtered_df["Region"].unique())
selected_regs = st.sidebar.multiselect(
    "Regions",
    options=all_regions,
    default=all_regions
)
if selected_regs:
    filtered_df = filtered_df[filtered_df["Region"].isin(selected_regs)]

st.sidebar.markdown("---")
st.sidebar.markdown("© AI Student Project | Deep Learning Visualization")
st.sidebar.caption(f"Showing {len(filtered_df)} orders from {start_date} to {end_date}")

# ---------------------------------------
# 3. MAIN LAYOUT WITH TABS
# ---------------------------------------
st.title("📊 Global Superstore Deep Learning Explorer")
tabs = st.tabs(["📈 Visualizations", "🧠 Model: No FE", "⚙️ Model: With FE + Tuner"])

# ---------------------------------------
# 3.1 TAB 1: VISUALIZATIONS
# ---------------------------------------
with tabs[0]:
    st.header("1. Exploratory Visualizations")

    # 1.1 KPI Cards (Total Sales, Total Profit, Total Orders, Avg Discount)
    total_sales = filtered_df["Sales"].sum()
    total_profit = filtered_df["Profit"].sum()
    total_orders = filtered_df.shape[0]
    avg_discount = filtered_df["Discount"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🛒 Total Orders", f"{total_orders:,}")
    col2.metric("💰 Total Sales (USD)", f"${total_sales:,.0f}")
    col3.metric("📈 Total Profit (USD)", f"${total_profit:,.0f}")
    col4.metric("🏷️ Avg Discount", f"{avg_discount:.1f}%")

    st.markdown("---")

    # 1.2 Time-Series: Sales & Profit Over Time
    st.subheader("Sales & Profit Over Time")
    temp = (
        filtered_df
        .assign(YearMonth=filtered_df["Order Date"].dt.to_period("M").dt.to_timestamp())
        .groupby("YearMonth")
        .agg({"Sales": "sum", "Profit": "sum"})
        .reset_index()
    )

    fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
    ax_ts.plot(temp["YearMonth"], temp["Sales"], marker="o", linestyle="-", color="tab:blue", label="Sales")
    ax_ts.plot(temp["YearMonth"], temp["Profit"], marker="o", linestyle="-", color="tab:green", label="Profit")
    ax_ts.set_xlabel("Month")
    ax_ts.set_ylabel("Amount (USD)")
    ax_ts.set_title("Monthly Sales & Profit")
    ax_ts.legend(loc="upper left")
    ax_ts.grid(alpha=0.3)
    fig_ts.autofmt_xdate()
    st.pyplot(fig_ts)

    st.markdown("---")

    # 1.3 Dynamic Histogram / Distribution
    st.subheader("Dynamic Distribution of Numeric Features")
    num_cols = ["Sales", "Profit", "Quantity", "Discount", "Shipping Cost"]
    chosen_num = st.selectbox("Select a Numeric Feature", options=num_cols, index=0)
    bin_count = st.slider("Number of Bins", min_value=20, max_value=100, value=40)

    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    sns.histplot(
        filtered_df[chosen_num],
        bins=bin_count,
        kde=True,
        ax=ax_hist,
        color="steelblue"
    )
    ax_hist.set_title(f"Distribution of {chosen_num}")
    ax_hist.set_xlabel(chosen_num)
    ax_hist.set_ylabel("Count")
    st.pyplot(fig_hist)

    st.markdown("---")

    # 1.4 Category & Sub-Category Breakdown (Side-by-side)
    st.subheader("Category & Sub-Category Breakdown")
    colA, colB = st.columns(2)

    with colA:
        sales_by_cat = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Sales", ascending=False)
        )
        fig_cat_sales, ax_cat_sales = plt.subplots(figsize=(6, 4))
        ax_cat_sales.barh(
            sales_by_cat["Category"],
            sales_by_cat["Sales"],
            color=plt.cm.Paired.colors
        )
        ax_cat_sales.invert_yaxis()
        ax_cat_sales.set_xlabel("Total Sales (USD)")
        ax_cat_sales.set_title("Total Sales by Category")
        for i, v in enumerate(sales_by_cat["Sales"]):
            ax_cat_sales.text(v + total_sales * 0.005, i, f"${v:,.0f}", va="center")
        st.pyplot(fig_cat_sales)

    with colB:
        profit_by_cat = (
            filtered_df.groupby("Category")["Profit"]
            .sum()
            .reset_index()
            .sort_values("Profit", ascending=False)
        )
        fig_cat_profit, ax_cat_profit = plt.subplots(figsize=(6, 4))
        ax_cat_profit.barh(
            profit_by_cat["Category"],
            profit_by_cat["Profit"],
            color=plt.cm.Paired.colors
        )
        ax_cat_profit.invert_yaxis()
        ax_cat_profit.set_xlabel("Total Profit (USD)")
        ax_cat_profit.set_title("Total Profit by Category")
        for i, v in enumerate(profit_by_cat["Profit"]):
            ax_cat_profit.text(v + total_profit * 0.005, i, f"${v:,.0f}", va="center")
        st.pyplot(fig_cat_profit)

    st.markdown("---")

    # 1.5 Scatter: Discount vs. Profit, colored by Category
    st.subheader("Discount vs. Profit (Colored by Category)")
    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
    categories = filtered_df["Category"].unique()
    palette = sns.color_palette("Set2", len(categories))
    for idx, cat in enumerate(categories):
        subset = filtered_df[filtered_df["Category"] == cat]
        ax_scatter.scatter(
            subset["Discount"],
            subset["Profit"],
            label=cat,
            alpha=0.6,
            s=30,
            color=palette[idx]
        )
    ax_scatter.set_xlabel("Discount")
    ax_scatter.set_ylabel("Profit (USD)")
    ax_scatter.set_title("How Discount Impacts Profit")
    ax_scatter.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_scatter.grid(alpha=0.3)
    st.pyplot(fig_scatter)

    st.markdown("---")

    # 1.6 Correlation Heatmap
    st.subheader("Correlation Matrix of Key Numerics")
    corr_df = filtered_df[["Sales", "Profit", "Quantity", "Discount", "Shipping Cost"]].corr().round(2)
    fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="RdBu",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax_corr
    )
    ax_corr.set_title("Correlation Heatmap")
    st.pyplot(fig_corr)

    st.markdown("---")

    # 1.7 Optional: Geographic Scatter (if lat/lon available)
    if "Latitude" in filtered_df.columns and "Longitude" in filtered_df.columns:
        st.subheader("Geographic Distribution of Sales")
        map_df = filtered_df[["Latitude", "Longitude", "Sales"]].copy()
        # Normalize bubble sizes
        min_sales = map_df["Sales"].min()
        max_sales = map_df["Sales"].max()
        map_df["size"] = ((map_df["Sales"] - min_sales) / (max_sales - min_sales)) * 200 + 20

        fig_map, ax_map = plt.subplots(figsize=(8, 5))
        sc = ax_map.scatter(
            map_df["Longitude"],
            map_df["Latitude"],
            s=map_df["size"],
            c=map_df["Sales"],
            cmap="viridis",
            alpha=0.6
        )
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Sales by Location (Bubble Size ∝ Sales)")
        cbar = plt.colorbar(sc, ax=ax_map)
        cbar.ax.set_ylabel("Sales (USD)")
        st.pyplot(fig_map)

# ---------------------------------------
# 3.2 TAB 2: MODEL WITHOUT FEATURE ENGINEERING
# ---------------------------------------
with tabs[1]:
    st.header("2. Model: No Feature Engineering")
    st.markdown(
        """
        In this section, we train a simple feed-forward neural network using only basic preprocessing:
        - Encode categorical variables via one-hot.
        - Scale numeric columns with StandardScaler.
        - Use “Order Date” as a raw timestamp.
        - Target: `Profit` (regression).
        """
    )

    st.markdown("#### 2.1 Data Preparation")
    st.write(
        """
        We drop non-predictive IDs and missing entries, convert `Order Date` to UNIX timestamp,
        one-hot encode categorical columns, and scale numeric features. Then, we split 80/20.
        """
    )

    if st.button("▶️ Prepare & Train Baseline Model", key="train_no_fe"):
        with st.spinner("Training baseline model… this may take ~30–60 seconds"):
            # 1. Prepare a copy so we don’t disturb the global filtered_df
            df0 = filtered_df.copy().dropna(
                subset=["Sales", "Profit", "Quantity", "Discount", "Shipping Cost"]
            )
            # Drop identifiers
            df0 = df0.drop(
                columns=[
                    "Row ID", "Order ID", "Customer ID", "Customer Name",
                    "Postal Code", "Product ID", "Product Name", "Ship Date"
                ],
                errors="ignore"
            )

            # Convert Order Date to timestamp
            df0["Order_TS"] = df0["Order Date"].astype(np.int64) // 10**9
            df0 = df0.drop(columns=["Order Date"])

            # One-hot encode categoricals
            cat_cols_0 = [
                "Ship Mode", "Segment", "City", "State", "Country",
                "Market", "Region", "Category", "Sub-Category", "Order Priority"
            ]
            df0 = pd.get_dummies(df0, columns=cat_cols_0, drop_first=True)

            TARGET = "Profit"
            X0 = df0.drop(columns=[TARGET]).values
            y0 = df0[TARGET].values.reshape(-1, 1)

            X_train0, X_test0, y_train0, y_test0 = train_test_split(
                X0, y0, test_size=0.2, random_state=42
            )

            # Identify numeric columns in df0
            numeric_cols0 = ["Sales", "Quantity", "Discount", "Shipping Cost", "Order_TS"]
            idx_nums0 = [df0.columns.get_loc(c) for c in numeric_cols0]
            scaler0 = StandardScaler()
            scaler0.fit(X_train0[:, idx_nums0])
            X_train0[:, idx_nums0] = scaler0.transform(X_train0[:, idx_nums0])
            X_test0[:, idx_nums0] = scaler0.transform(X_test0[:, idx_nums0])

            # Build baseline model
            input_dim0 = X_train0.shape[1]
            model0 = keras.Sequential([
                keras.layers.InputLayer(input_shape=(input_dim0,)),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1, activation="linear")
            ])
            model0.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss="mse",
                metrics=["mae"]
            )

            # Train for a user-selected number of epochs
            epochs0 = st.number_input(
                "Epochs (Baseline)", min_value=5, max_value=100, value=30, step=5, key="epochs0"
            )
            history0 = model0.fit(
                X_train0, y_train0,
                validation_split=0.1,
                epochs=int(epochs0),
                batch_size=32,
                verbose=0
            )

            # Show training curves with Matplotlib
            train_mae = history0.history["mae"]
            val_mae = history0.history["val_mae"]
            epochs_range = list(range(1, len(train_mae) + 1))
            fig_train, ax_train = plt.subplots(figsize=(8, 4))
            ax_train.plot(epochs_range, train_mae, marker="o", label="Train MAE")
            ax_train.plot(epochs_range, val_mae, marker="o", label="Val MAE")
            ax_train.set_xlabel("Epoch")
            ax_train.set_ylabel("MAE")
            ax_train.set_title("MAE vs. Epochs (Baseline)")
            ax_train.legend()
            ax_train.grid(alpha=0.3)
            st.pyplot(fig_train)

            # Evaluate on test set
            test_loss0, test_mae0 = model0.evaluate(X_test0, y_test0, verbose=0)
            st.success(f"✅ Baseline Test MAE: {test_mae0:.2f} USD")

    st.markdown(
        """
        *Tip: If training is slow, you can reduce “Epochs (Baseline)” or sample a smaller set of rows in `filtered_df`.*  
        """
    )

# ---------------------------------------
# 3.3 TAB 3: MODEL WITH FEATURE ENGINEERING & KERAS TUNER
# ---------------------------------------
with tabs[2]:
    st.header("3. Model: Feature Engineering + Keras Tuner")
    st.markdown(
        """
        Here, we perform extensive feature engineering—date parts, ratios, flags, grouping high-cardinality categories—then use **Keras Tuner** 
        to search for the best network architecture (layers, units, dropout, learning rate).
        """
    )

    st.markdown("#### 3.1 Feature Engineering Steps")
    st.markdown(
        """
        1. **Date Parts**: Extract `order_year`, `order_month`, `order_dow` (day of week), 
           `order_week`, `order_is_weekend`.  
        2. **Shipping Time**: `(Ship Date − Order Date).days`.  
        3. **Profit Margin**: `Profit / Sales`.  
        4. **Price per Unit**: `Sales / Quantity`.  
        5. **High Discount Flag**: `Discount > 20%`.  
        6. **Group “City” & “State”**: Keep top-20 frequent, rest → “Other”.  
        7. One-hot encode medium-cardinality categoricals.  
        8. Scale all numeric features.  
        """
    )

    if st.button("▶️ Run FE + Hyperparameter Search", key="run_fe_tuner"):
        with st.spinner("Feature engineering & hyperparameter tuning… this may take a few minutes"):
            # 3.2 Deep Copy & FE
            df1 = filtered_df.copy()

            # Drop rows with missing in key columns
            df1 = df1.dropna(
                subset=[
                    "Sales", "Profit", "Quantity", "Discount", "Shipping Cost",
                    "Order Date", "Ship Date"
                ]
            )

            # Drop identifiers
            df1 = df1.drop(
                columns=[
                    "Row ID", "Order ID", "Customer ID", "Customer Name",
                    "Postal Code", "Product ID", "Product Name"
                ],
                errors="ignore"
            )

            # Dates to features
            df1["order_year"] = df1["Order Date"].dt.year
            df1["order_month"] = df1["Order Date"].dt.month
            df1["order_dow"] = df1["Order Date"].dt.dayofweek
            df1["order_week"] = df1["Order Date"].dt.isocalendar().week.astype(int)
            df1["order_is_weekend"] = df1["order_dow"].isin([5, 6]).astype(int)

            # Shipping time
            df1["shipping_time"] = (df1["Ship Date"] - df1["Order Date"]).dt.days

            # Profit margin
            df1["profit_margin"] = df1["Profit"] / df1["Sales"]
            df1["profit_margin"] = df1["profit_margin"].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Price per unit
            df1["price_per_unit"] = df1["Sales"] / df1["Quantity"].replace(0, np.nan)
            df1["price_per_unit"] = df1["price_per_unit"].replace([np.inf, -np.inf], np.nan).fillna(0)

            # High discount flag
            df1["high_discount_flag"] = (df1["Discount"] > 0.2).astype(int)

            # Drop original dates
            df1 = df1.drop(columns=["Order Date", "Ship Date"])

            # Group “City” top-20, rest = “Other”
            top_cities = df1["City"].value_counts().nlargest(20).index
            df1["City_FE"] = df1["City"].where(df1["City"].isin(top_cities), other="Other")
            df1 = df1.drop(columns=["City"])

            # Group “State” top-20, rest = “Other”
            top_states = df1["State"].value_counts().nlargest(20).index
            df1["State_FE"] = df1["State"].where(df1["State"].isin(top_states), other="Other")
            df1 = df1.drop(columns=["State"])

            # Select final features
            feature_cols1 = [
                "Sales", "Quantity", "Discount", "Shipping Cost",
                "order_year", "order_month", "order_dow", "order_week",
                "order_is_weekend", "shipping_time", "profit_margin", "price_per_unit",
                "high_discount_flag",
                "Ship Mode", "Segment", "Category", "Sub-Category",
                "Market", "Region", "Order Priority", "City_FE", "State_FE", "Country"
            ]
            df1 = df1[feature_cols1 + ["Profit"]].dropna(axis=0, how="any")
            
            # 3.3 Split X / y
            TARGET = "Profit"
            y1 = df1[TARGET].values.reshape(-1, 1)
            X1_df = df1.drop(columns=[TARGET])

            X1_train_df, X1_test_df, y1_train, y1_test = train_test_split(
                X1_df, y1, test_size=0.2, random_state=42
            )

            # 3.4 Separate Numeric & Categorical
            num_cols1 = [
                "Sales", "Quantity", "Discount", "Shipping Cost",
                "order_year", "order_month", "order_dow", "order_week",
                "order_is_weekend", "shipping_time", "profit_margin", "price_per_unit",
                "high_discount_flag"
            ]
            cat_cols1 = [
                "Ship Mode", "Segment", "Category", "Sub-Category",
                "Market", "Region", "Order Priority", "City_FE", "State_FE", "Country"
            ]

            # Scale numerics
            scaler1 = StandardScaler()
            X1_train_nums = scaler1.fit_transform(X1_train_df[num_cols1])
            X1_test_nums = scaler1.transform(X1_test_df[num_cols1])

            # OneHot encode categorical
            ohe1 = OneHotEncoder(sparse=False, handle_unknown="ignore")
            X1_train_cats = ohe1.fit_transform(X1_train_df[cat_cols1])
            X1_test_cats = ohe1.transform(X1_test_df[cat_cols1])

            # Concatenate
            X1_train = np.hstack([X1_train_nums, X1_train_cats])
            X1_test = np.hstack([X1_test_nums, X1_test_cats])
            input_dim1 = X1_train.shape[1]

            # 3.5 Define Hypermodel for Keras Tuner
            def build_hypermodel(hp):
                model = keras.Sequential()
                model.add(keras.layers.InputLayer(input_shape=(input_dim1,)))

                # Tune number of layers (1–3)
                for i in range(hp.Int("num_layers", 1, 3)):
                    units = hp.Choice(f"units_{i}", [32, 64, 128])
                    model.add(keras.layers.Dense(units, activation="relu"))
                    # Tune dropout
                    drop = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
                    if drop > 0:
                        model.add(keras.layers.Dropout(drop))

                model.add(keras.layers.Dense(1, activation="linear"))

                lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=lr),
                    loss="mse",
                    metrics=["mae"]
                )
                return model

            # 3.6 Run Keras Tuner (Random Search by default)
            tuner = kt.RandomSearch(
                build_hypermodel,
                objective="val_mae",
                max_trials=8,
                executions_per_trial=2,
                directory="kt_dir",
                project_name="gs_superstore_fe"
            )

            early_stop_cb = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            tuner.search(
                X1_train,
                y1_train,
                validation_split=0.1,
                epochs=40,
                batch_size=32,
                callbacks=[early_stop_cb],
                verbose=0
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            st.success("✅ Hyperparameter search complete!")
            st.write("**Best Hyperparameters Found:**")
            st.write(f"- Number of layers: {best_hps.get('num_layers')}")
            for i in range(best_hps.get("num_layers")):
                st.write(f"  - Units in layer {i + 1}: {best_hps.get(f'units_{i}')}  (dropout: {best_hps.get(f'dropout_{i}'):0.2f})")
            st.write(f"- Learning Rate: {best_hps.get('learning_rate'):0.5f}")

            # 3.7 Retrain Best Model on Full Train Set
            best_model = tuner.get_best_models(num_models=1)[0]
            st.write("Retraining best model on full training set…")
            history1 = best_model.fit(
                X1_train, y1_train,
                validation_split=0.1,
                epochs=40,
                batch_size=32,
                callbacks=[early_stop_cb],
                verbose=0
            )

            # Plot training curves (Matplotlib)
            train_mae1 = history1.history["mae"]
            val_mae1 = history1.history["val_mae"]
            epochs1 = list(range(1, len(train_mae1) + 1))
            fig_train1, ax_train1 = plt.subplots(figsize=(8, 4))
            ax_train1.plot(epochs1, train_mae1, marker="o", label="Train MAE")
            ax_train1.plot(epochs1, val_mae1, marker="o", label="Val MAE")
            ax_train1.set_xlabel("Epoch")
            ax_train1.set_ylabel("MAE")
            ax_train1.set_title("MAE vs. Epochs (FE Model)")
            ax_train1.legend()
            ax_train1.grid(alpha=0.3)
            st.pyplot(fig_train1)

            # Evaluate on test set
            test_loss1, test_mae1 = best_model.evaluate(X1_test, y1_test, verbose=0)
            st.success(f"✅ Feature-Engineered Model Test MAE: {test_mae1:.2f} USD")

    st.markdown(
        """
        *Tip: You can modify the search space in `build_hypermodel` (e.g., try different unit sizes or add batch normalization).  
        Also, if the tuner runs too long, reduce `max_trials` or `epochs` in the tuner search call.*  
        """
    )
