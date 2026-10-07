import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page layout & configuration
st.set_page_config(
    page_title="FIT & FINE - Business Analytics Dashboard",
    page_icon="👔",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 30px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">FIT & FINE (SAMBALPUR) — RETAIL ANALYTICS SUITE</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fit_and_fine_sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Total_Revenue"] = df["Quantity"] * df["Selling_Price_INR"]
    df["Total_Cost"] = df["Quantity"] * df["Cost_Price_INR"]
    df["Gross_Profit"] = df["Total_Revenue"] - df["Total_Cost"]
    return df

df = load_data()

# ------------------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------------------
st.sidebar.header("Filter Transactions")
selected_category = st.sidebar.multiselect(
    "Select Product Categories:",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

selected_payment = st.sidebar.multiselect(
    "Select Payment Methods:",
    options=df["Payment_Method"].unique(),
    default=df["Payment_Method"].unique()
)

filtered_df = df[
    (df["Category"].isin(selected_category)) & 
    (df["Payment_Method"].isin(selected_payment))
]

# ------------------------------------------------------------------------------
# TOP METRICS CARD DISPLAY
# ------------------------------------------------------------------------------
total_rev = filtered_df["Total_Revenue"].sum()
total_profit = filtered_df["Gross_Profit"].sum()
total_units = filtered_df["Quantity"].sum()
margin = (total_profit / total_rev * 100) if total_rev > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{total_rev:,.2f}")
col2.metric("Gross Profit", f"₹{total_profit:,.2f}")
col3.metric("Profit Margin", f"{margin:.1f}%")
col4.metric("Total Units Sold", f"{total_units:,}")

st.markdown("---")

# ------------------------------------------------------------------------------
# TAB NAVIGATION FOR THE 3 SERVICES
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Module 1: Inventory & Dead-Stock", 
    "👥 Module 2: RFM Customer Segmentation", 
    "📈 Module 3: Executive Dashboard & Trends"
])

# ==============================================================================
# TAB 1: INVENTORY & DEAD-STOCK OPTIMIZATION
# ==============================================================================
with tab1:
    st.subheader("Inventory Movement & Stock Health Classification")
    
    inv_summary = filtered_df.groupby(["Category", "Size"]).agg(
        Total_Units_Sold=("Quantity", "sum"),
        Revenue_Generated=("Total_Revenue", "sum"),
        Last_Sale_Date=("Date", "max")
    ).reset_index()

    max_date = filtered_df["Date"].max()
    inv_summary["Days_Unsold"] = (max_date - inv_summary["Last_Sale_Date"]).dt.days

    def classify_stock(row):
        if row["Total_Units_Sold"] >= 30:
            return "Fast-Moving (Reorder)"
        elif row["Total_Units_Sold"] >= 10:
            return "Regular Movement"
        elif row["Days_Unsold"] > 90 or row["Total_Units_Sold"] < 5:
            return "Dead-Stock (Clearance)"
        else:
            return "Slow-Moving"

    inv_summary["Stock_Status"] = inv_summary.apply(classify_stock, axis=1)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("##### Stock Status Breakdown")
        st.dataframe(inv_summary["Stock_Status"].value_counts(), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = inv_summary["Stock_Status"].value_counts()
        ax.bar(counts.index, counts.values, color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"])
        ax.set_title("SKU Health Distribution")
        st.pyplot(fig)

    st.write("##### Detailed Inventory Action Plan")
    st.dataframe(inv_summary, use_container_width=True)

# ==============================================================================
# TAB 2: CUSTOMER PURCHASE SEGMENTATION (RFM)
# ==============================================================================
with tab2:
    st.subheader("RFM Customer Segmentation & Target Marketing")
    
    analysis_date = filtered_df["Date"].max() + pd.Timedelta(days=1)
    
    rfm = filtered_df.groupby("Customer_ID").agg({
        "Date": lambda x: (analysis_date - x.max()).days,
        "Invoice_ID": "nunique",
        "Total_Revenue": "sum"
    }).reset_index()

    rfm.columns = ["Customer_ID", "Recency", "Frequency", "Monetary"]

    rfm["R_Score"] = pd.qcut(rfm["Recency"], 4, labels=[4, 3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 4, labels=[1, 2, 3, 4])

    rfm["RFM_Cell"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)

    def segment_customer(row):
        if row["RFM_Cell"] in ["444", "443", "344"]:
            return "VIP Champions"
        elif row["F_Score"] in [3, 4]:
            return "Loyal Regulars"
        elif row["R_Score"] == 4:
            return "Recent Buyers"
        elif row["R_Score"] in [1, 2] and row["M_Score"] in [3, 4]:
            return "At-Risk High Value"
        else:
            return "Low-Engagement Casuals"

    rfm["Customer_Segment"] = rfm.apply(segment_customer, axis=1)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("##### Segment Share")
        st.dataframe(rfm["Customer_Segment"].value_counts(), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        seg_counts = rfm["Customer_Segment"].value_counts()
        ax.pie(seg_counts.values, labels=seg_counts.index, autopct="%1.1f%%", startangle=140)
        ax.set_title("Customer Base Distribution")
        st.pyplot(fig)

    st.write("##### Target WhatsApp Marketing Segment Table")
    st.dataframe(rfm[["Customer_ID", "Recency", "Frequency", "Monetary", "Customer_Segment"]], use_container_width=True)

# ==============================================================================
# TAB 3: EXECUTIVE REPORTING DASHBOARD
# ==============================================================================
with tab3:
    st.subheader("Revenue & Profitability Visualizations")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("##### Monthly Revenue & Profit Trend")
        filtered_df["YearMonth"] = filtered_df["Date"].dt.to_period("M")
        monthly = filtered_df.groupby("YearMonth")[["Total_Revenue", "Gross_Profit"]].sum()
        monthly.index = monthly.index.astype(str)
        
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(monthly.index, monthly["Total_Revenue"], marker='o', label="Revenue", color="#1f77b4")
        ax.plot(monthly.index, monthly["Gross_Profit"], marker='s', label="Profit", color="#2ca02c")
        ax.set_xticklabels(monthly.index, rotation=45)
        ax.legend()
        st.pyplot(fig)

    with col_b:
        st.write("##### Revenue Contribution by Category")
        cat_rev = filtered_df.groupby("Category")["Total_Revenue"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(cat_rev.index, cat_rev.values, color="#ff7f0e")
        st.pyplot(fig)
