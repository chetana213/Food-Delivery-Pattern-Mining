import streamlit as st
import pandas as pd

from dataset_module import (
    load_dataset,
    get_dataset_overview,
    get_missing_values,
    get_data_types,
    get_dataset_preview
)

from geographic_module import (
    calculate_delivery_distance,
    get_distance_dataset,
    get_distance_statistics,
    get_distance_time_correlation,
    create_distance_time_plot
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Pattern Mining",
    page_icon="🍽️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🍽️ Food Delivery Pattern Mining")

st.markdown(
    """
    ### Data & Geographic Analysis

    This section presents the dataset overview, data quality analysis,
    and geographic distance analysis of the food delivery dataset.
    """
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return load_dataset("Zomato Dataset.csv")


df = load_data()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("1. Dataset Overview")

overview = get_dataset_overview(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Records",
        f"{overview['rows']:,}"
    )

with col2:
    st.metric(
        "Attributes",
        overview["columns"]
    )

with col3:
    st.metric(
        "Duplicate Records",
        overview["duplicates"]
    )

with col4:
    st.metric(
        "Average Delivery Time",
        f"{df['Time_taken (min)'].mean():.2f} min"
    )


# ============================================================
# DATASET PERIOD
# ============================================================

st.subheader("Dataset Period")

date_col1, date_col2 = st.columns(2)

with date_col1:
    st.metric(
        "Start Date",
        overview["start_date"].strftime("%d-%b-%Y")
    )

with date_col2:
    st.metric(
        "End Date",
        overview["end_date"].strftime("%d-%b-%Y")
    )


# ============================================================
# DATASET PREVIEW
# ============================================================

st.subheader("Dataset Preview")

preview_rows = st.slider(
    "Number of rows to display",
    min_value=5,
    max_value=20,
    value=10
)

st.dataframe(
    get_dataset_preview(df, preview_rows),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()

st.header("2. Data Quality Analysis")

missing_values = get_missing_values(df)

missing_table = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

missing_table["Missing Percentage"] = (
    missing_table["Missing Values"]
    / len(df)
    * 100
).round(2)

missing_table = missing_table.sort_values(
    "Missing Values",
    ascending=False
)

st.subheader("Missing Values")

st.dataframe(
    missing_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DATA TYPES
# ============================================================

with st.expander("View Data Types"):

    dtype_table = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        dtype_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

st.subheader("Data Quality Summary")

quality_col1, quality_col2, quality_col3 = st.columns(3)

with quality_col1:
    st.metric(
        "Total Records",
        f"{len(df):,}"
    )

with quality_col2:
    st.metric(
        "Duplicate Records",
        int(df.duplicated().sum())
    )

with quality_col3:
    st.metric(
        "Columns with Missing Values",
        int((missing_values > 0).sum())
    )


# ============================================================
# GEOGRAPHIC ANALYSIS
# ============================================================

st.divider()

st.header("3. Geographic Analysis")

st.markdown(
    """
    Geographic distance between the restaurant and delivery location
    is calculated using the **Haversine formula** based on latitude
    and longitude coordinates.
    """
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("View Geographic Methodology"):

    st.markdown(
        """
        **Distance calculation method**

        - Input 1: Restaurant latitude and longitude
        - Input 2: Delivery location latitude and longitude
        - Distance calculation: Haversine formula
        - Earth radius used: 6,371 km
        - Distance validation threshold: 50 km

        The Haversine formula calculates the approximate great-circle
        distance between two geographic coordinates.
        """
    )


# ============================================================
# CALCULATE DISTANCE
# ============================================================

df_with_distance = calculate_delivery_distance(df)

df_distance = get_distance_dataset(
    df_with_distance,
    max_distance=50
)


# ============================================================
# DISTANCE STATISTICS
# ============================================================

distance_stats = get_distance_statistics(df_distance)

excluded_records = (
    len(df_with_distance) - len(df_distance)
)


geo_col1, geo_col2, geo_col3, geo_col4 = st.columns(4)

with geo_col1:
    st.metric(
        "Valid Distance Records",
        f"{distance_stats['count']:,}"
    )

with geo_col2:
    st.metric(
        "Average Distance",
        f"{distance_stats['mean']:.2f} km"
    )

with geo_col3:
    st.metric(
        "Median Distance",
        f"{distance_stats['median']:.2f} km"
    )

with geo_col4:
    st.metric(
        "Excluded Records",
        f"{excluded_records:,}"
    )


# ============================================================
# DISTANCE RANGE
# ============================================================

range_col1, range_col2 = st.columns(2)

with range_col1:
    st.metric(
        "Minimum Distance",
        f"{distance_stats['min']:.2f} km"
    )

with range_col2:
    st.metric(
        "Maximum Distance",
        f"{distance_stats['max']:.2f} km"
    )


# ============================================================
# DISTANCE VALIDATION
# ============================================================

st.subheader("Distance Validation")

st.info(
    f"""
    **{excluded_records:,} records were excluded from distance-based
    analysis.**

    Their calculated geographic distance exceeded the 50 km validation
    threshold. These records were excluded because their coordinates
    produced unusually large distances that were inconsistent with
    the delivery-distance range observed in the dataset.

    The original records remain in the dataset and are not removed
    from other analyses.
    """
)


validation_col1, validation_col2, validation_col3 = st.columns(3)

with validation_col1:
    st.metric(
        "Original Records",
        f"{len(df_with_distance):,}"
    )

with validation_col2:
    st.metric(
        "Valid Records",
        f"{len(df_distance):,}"
    )

with validation_col3:
    st.metric(
        "Excluded Records",
        f"{excluded_records:,}"
    )


# ============================================================
# DISTANCE VS DELIVERY TIME
# ============================================================

st.subheader("Delivery Distance vs Delivery Time")

correlation = get_distance_time_correlation(
    df_distance
)

st.plotly_chart(
    create_distance_time_plot(df_distance),
    use_container_width=True
)


# ============================================================
# CORRELATION FINDING
# ============================================================

st.subheader("Geographic Analysis Finding")

st.metric(
    "Pearson Correlation",
    f"{correlation:.3f}"
)

st.markdown(
    f"""
    The Pearson correlation coefficient between delivery distance
    and delivery time is **{correlation:.3f}**.

    This indicates a **moderate positive association** between the
    two variables. Longer delivery distances tend to be associated
    with longer delivery times, although distance alone does not
    explain the complete variation in delivery time.
    """
)


# ============================================================
# DETAILED DISTANCE STATISTICS
# ============================================================

with st.expander("View Detailed Distance Statistics"):

    statistics_table = pd.DataFrame({
        "Statistic": [
            "Number of valid records",
            "Mean distance",
            "Median distance",
            "Minimum distance",
            "Maximum distance"
        ],
        "Value": [
            f"{distance_stats['count']:,}",
            f"{distance_stats['mean']:.2f} km",
            f"{distance_stats['median']:.2f} km",
            f"{distance_stats['min']:.2f} km",
            f"{distance_stats['max']:.2f} km"
        ]
    })

    st.table(statistics_table)


# ============================================================
# DOWNLOAD DATASET
# ============================================================

st.subheader("Export Geographic Analysis Data")

csv_data = df_distance.to_csv(index=False)

st.download_button(
    label="Download Valid Geographic Dataset",
    data=csv_data,
    file_name="food_delivery_geographic_analysis.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Food Delivery Pattern Mining • Member 1: Data & Geographic Analysis"
)