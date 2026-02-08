import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Minimal styling (only header + KPI cards, no global colors)
# ---------------------------------------------------------
def add_custom_style():
    st.markdown(
        """
        <style>
        /* Header card */
        .header-card {
            padding: 1.2rem 1.5rem;
            border-radius: 0.9rem;
            background: linear-gradient(90deg, #e0f2fe, #dcfce7);
            border: 1px solid #e5e7eb;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
        }
        /* "Sales Data Analyzer" text */
        .header-card h1 {
            font-size: 1.9rem;
            margin: 0;
            color: #111827 !important;  /* dark heading */
        }
        /* "Upload large CSV/Excel..." text */
        .header-card p {
            margin: 0.3rem 0 0 0;
            color: #111827 !important;  /* dark description */
            font-size: 0.95rem;
        }

        /* KPI cards */
        .kpi-card {
            background: #ffffff;
            border-radius: 0.9rem;
            padding: 0.9rem 1.1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.6rem;
        }
        .kpi-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }
        .kpi-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #111827;
        }
        .kpi-sub {
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(label: str, value: str, sub: str = "", border_color: str = "#2563eb"):
    return f"""
    <div class="kpi-card" style="border-top: 3px solid {border_color};">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


add_custom_style()


# ---------------------------------------------------------
# Helper functions (data)
# ---------------------------------------------------------
def detect_column(df, candidates):
    """Find a column whose name contains one of the candidate strings (case-insensitive)."""
    for cand in candidates:
        for c in df.columns:
            if cand in c.lower():
                return c
    return None


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize / clean data and add revenue + date fields."""
    df = df.copy()

    col_date = detect_column(df, ["order_date", "date"])
    col_region = detect_column(df, ["region"])
    col_product = detect_column(df, ["product", "item", "sku"])
    col_qty = detect_column(df, ["quantity", "qty", "units"])
    col_price = detect_column(df, ["unit_price", "unit price", "price"])

    required = {
        "order_date": col_date,
        "region": col_region,
        "product": col_product,
        "quantity": col_qty,
        "unit_price": col_price,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required columns (or unable to auto-detect): {missing}. "
            "Make sure the file has date, region, product, quantity, and unit price."
        )

    df = df.rename(
        columns={
            col_date: "order_date",
            col_region: "region",
            col_product: "product",
            col_qty: "quantity",
            col_price: "unit_price",
        }
    )

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    df = df.dropna(subset=["order_date", "region", "product", "quantity", "unit_price"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    df = df.drop_duplicates()

    df["revenue"] = df["quantity"] * df["unit_price"]
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.to_period("M").astype(str)

    return df


def generate_reports(df: pd.DataFrame):
    """Return a dict of summary dataframes."""
    reports = {}

    total_revenue = df["revenue"].sum()
    total_quantity = df["quantity"].sum()
    num_orders = len(df)
    avg_order_value = total_revenue / num_orders if num_orders > 0 else 0

    kpi_df = pd.DataFrame(
        {
            "Metric": [
                "Total Revenue",
                "Total Quantity Sold",
                "Number of Orders",
                "Average Order Value",
            ],
            "Value": [total_revenue, total_quantity, num_orders, avg_order_value],
        }
    )
    reports["overall_kpis"] = kpi_df

    monthly = (
        df.groupby("month", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        .sort_values("month")
    )
    reports["monthly_performance"] = monthly

    by_region = (
        df.groupby("region", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
    )
    reports["revenue_by_region"] = by_region

    by_product = (
        df.groupby("product", as_index=False)
        .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
    )
    reports["revenue_by_product"] = by_product

    pivot_region_month = pd.pivot_table(
        df,
        index="region",
        columns="month",
        values="revenue",
        aggfunc="sum",
        fill_value=0,
    )
    reports["region_month_pivot"] = pivot_region_month

    return reports


def filter_data(df, date_range=None, regions=None, products=None):
    mask = pd.Series(True, index=df.index)

    if date_range is not None and len(date_range) == 2:
        start, end = date_range
        if start:
            mask &= df["order_date"] >= pd.to_datetime(start)
        if end:
            mask &= df["order_date"] <= pd.to_datetime(end)

    if regions:
        mask &= df["region"].isin(regions)

    if products:
        mask &= df["product"].isin(products)

    return df[mask]


def create_excel_report(reports_dict):
    """Create a multi-sheet Excel file in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, data in reports_dict.items():
            sheet_name = name[:31].replace(" ", "_").replace("/", "_")
            index_flag = not isinstance(data.index, pd.RangeIndex)
            data.to_excel(writer, sheet_name=sheet_name, index=index_flag)
    output.seek(0)
    return output


# ---------------------------------------------------------
# App layout
# ---------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <h1>📊 Sales Data Analyzer</h1>
        <p>Upload large CSV/Excel sales files, clean them, explore KPIs, and export Excel reports — all in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Tip: To allow files >200MB, change `maxUploadSize` in `.streamlit/config.toml` "
    "(for example, `1024` for 1 GB)."
)

uploaded_file = st.file_uploader(
    "Upload a sales file (CSV or Excel)", 
    type=["csv", "xlsx", "xls"],
    help="The app will automatically detect columns like date, region, product, quantity, and unit price.",
)

if uploaded_file is None:
    st.info("Upload a file to start the analysis.")
    st.stop()

# Read file
with st.spinner("Reading file..."):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

if "raw_df" in locals():
    with st.expander("Raw data (first 50 rows)"):
        st.dataframe(raw_df.head(50), use_container_width=True)
else:
    st.info("Please load data to view raw table")

# Clean
with st.spinner("Cleaning and preparing data..."):
    try:
        df = clean_sales_data(raw_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while cleaning data: {e}")
        st.stop()

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")

    min_date, max_date = df["order_date"].min(), df["order_date"].max()
    date_range = st.date_input(
        "Order date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    regions = sorted(df["region"].unique())
    selected_regions = st.multiselect("Regions", options=regions, default=regions)

    products = sorted(df["product"].unique())
    selected_products = st.multiselect(
        "Products (optional)", options=products, default=[]
    )

    st.markdown("---")
    st.caption("Use filters to narrow down the view. Helpful for very large files.")

filtered_df = filter_data(
    df,
    date_range=date_range if isinstance(date_range, (list, tuple)) else None,
    regions=selected_regions,
    products=selected_products if selected_products else None,
)

if filtered_df.empty:
    st.warning("No data matches the selected filters. Adjust filters to see results.")
    st.stop()

reports = generate_reports(filtered_df)


# Tabs
tab_overview, tab_insights, tab_export = st.tabs(
    ["📁 Data", "📈 Insights", "📤 Export"]
)

# ---------------------------------------------------------
# Tab: Data
# ---------------------------------------------------------
with tab_overview:
    st.subheader("Filtered data (preview)")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    st.markdown("#### Dataset summary")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Rows after filtering: **{len(filtered_df):,}**")
        st.write(f"Distinct products: **{filtered_df['product'].nunique():,}**")
    with c2:
        st.write(f"Distinct regions: **{filtered_df['region'].nunique():,}**")
        st.write(
            f"Date range: **{filtered_df['order_date'].min().date()}** → "
            f"**{filtered_df['order_date'].max().date()}**"
        )

# ---------------------------------------------------------
# Tab: Insights
# ---------------------------------------------------------
with tab_insights:
    st.subheader("Key performance indicators")

    kpi_df = reports["overall_kpis"]
    total_rev = float(
        kpi_df.loc[kpi_df["Metric"] == "Total Revenue", "Value"].iloc[0]
    )
    total_qty = float(
        kpi_df.loc[kpi_df["Metric"] == "Total Quantity Sold", "Value"].iloc[0]
    )
    num_orders = int(
        kpi_df.loc[kpi_df["Metric"] == "Number of Orders", "Value"].iloc[0]
    )
    aov = float(
        kpi_df.loc[kpi_df["Metric"] == "Average Order Value", "Value"].iloc[0]
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            kpi_card(
                "Total Revenue",
                f"${total_rev:,.0f}",
                "Sum of all sales in the filtered range.",
                "#22c55e",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card(
                "Total Quantity",
                f"{total_qty:,.0f}",
                "Total units sold.",
                "#3b82f6",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card(
                "Number of Orders",
                f"{num_orders:,}",
                "Total order rows.",
                "#a855f7",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                "Avg. Order Value",
                f"${aov:,.2f}",
                "Revenue per order row.",
                "#f97316",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("### Trends & breakdown")

    col_l, col_r = st.columns(2)

    # Monthly revenue chart
    with col_l:
        st.markdown("**Monthly revenue**")
        monthly = reports["monthly_performance"]
        if not monthly.empty:
            chart_df = monthly.set_index("month")["revenue"]
            st.bar_chart(chart_df, use_container_width=True)
        else:
            st.write("No data for the selected filters.")

    # Region revenue chart
    with col_r:
        st.markdown("**Revenue by region**")
        by_region = reports["revenue_by_region"]
        if not by_region.empty:
            chart_df = by_region.set_index("region")["revenue"]
            st.bar_chart(chart_df, use_container_width=True)
        else:
            st.write("No data for the selected filters.")

    st.markdown("### Top products by revenue")
    st.dataframe(
        reports["revenue_by_product"].head(20),
        use_container_width=True,
    )

    st.markdown("### Region × month revenue (pivot)")
    st.dataframe(reports["region_month_pivot"], use_container_width=True)

# ---------------------------------------------------------
# Tab: Export
# ---------------------------------------------------------
with tab_export:
    st.subheader("Export analytical summaries")

    excel_buffer = create_excel_report(reports)
    st.download_button(
        label="📥 Download Excel report",
        data=excel_buffer,
        file_name="sales_analysis_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown("#### Download filtered data as CSV")
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=csv_data,
        file_name="filtered_sales_data.csv",
        mime="text/csv",
    )