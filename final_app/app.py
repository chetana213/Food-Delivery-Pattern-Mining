import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Food Delivery Pattern Mining",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from member1.dataset_module import (
    load_dataset,
    get_dataset_overview,
    get_missing_values,
    get_data_types,
    get_dataset_preview
)

from member1.geographic_module import (
    calculate_delivery_distance,
    get_distance_dataset,
    get_distance_statistics,
    get_distance_time_correlation,
    create_distance_time_plot
)

from member2.temporal_module import (
    prepare_temporal_features,
    get_hourly_delivery_pattern,
    get_day_of_week_pattern,
    get_weekend_pattern,
    create_hourly_delivery_plot,
    create_day_of_week_plot
)

from member2.operational_module import (
    get_traffic_pattern
)

from member3.clustering_module import (
    prepare_clustering_data,
    scale_features,
    calculate_elbow_values,
    calculate_silhouette_values,
    create_elbow_plot,
    create_silhouette_plot,
    perform_kmeans,
    get_cluster_profile,
    create_cluster_time_plot,
    create_cluster_size_plot,
    get_traffic_profile,
    get_weather_profile,
    get_vehicle_profile,
    create_traffic_heatmap,
    create_weather_heatmap,
    create_vehicle_heatmap
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.3rem;
    }

    .sidebar-subtitle {
        color: #667085;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }

    /* Hero section */
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #f7f9fc 0%,
            #eef4fb 100%
        );
        border: 1px solid #e5eaf1;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 750;
        color: #172033;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #475467;
        margin-bottom: 0;
    }

    /* Section heading */
    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #172033;
        margin-top: 1.4rem;
        margin-bottom: 0.8rem;
    }

    .section-description {
        color: #667085;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* Information cards */
    .info-card {
        padding: 1.2rem;
        border-radius: 14px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        min-height: 145px;
    }

    .info-card-title {
        font-weight: 700;
        color: #172033;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .info-card-text {
        color: #667085;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Finding cards */
    .finding-card {
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e6eaf0;
        margin-bottom: 0.7rem;
    }

    .finding-title {
        font-weight: 650;
        color: #172033;
        margin-bottom: 0.2rem;
    }

    .finding-text {
        color: #667085;
        font-size: 0.9rem;
    }

    /* Methodology */
    .method-card {
        text-align: center;
        padding: 1.1rem;
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e6eaf0;
        min-height: 115px;
    }

    .method-number {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.3rem;
    }

    .method-name {
        font-weight: 650;
        color: #172033;
    }

    /* Small label */
    .small-label {
        color: #667085;
        font-size: 0.82rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 0.8rem;
        padding-top: 2rem;
        padding-bottom: 0.5rem;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA PATH
# =========================================================

DATA_PATH = PROJECT_ROOT / "member1" / "Zomato Dataset.csv"


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return load_dataset(DATA_PATH)


df = load_data()


# =========================================================
# PREPARE COMMON DATA
# =========================================================

@st.cache_data
def prepare_common_data(data):
    temporal_df = prepare_temporal_features(data)
    geographic_df = calculate_delivery_distance(temporal_df)
    return geographic_df


df_prepared = prepare_common_data(df)


# =========================================================
# CLUSTERING DATA
# =========================================================

@st.cache_data
def prepare_cluster_data(data):
    return prepare_clustering_data(data)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🍔 Food Delivery Mining</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Data Mining Analytics Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Navigation")

    page = st.radio(
        "Select Analysis",
        [
            "Home",
            "Dataset Overview",
            "Geographic Analysis",
            "Temporal Analysis",
            "Operational Analysis",
            "K-Means Clustering"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="small-label">
        <b>Project</b><br>
        Food Delivery Pattern Mining
        <br><br>

        <b>Objective</b><br>
        Discover temporal, geographical,
        environmental and operational
        patterns in food delivery data.
        <br><br>

        <b>Techniques</b><br>
        • Data preprocessing<br>
        • Temporal analysis<br>
        • Geographic analysis<br>
        • Operational analysis<br>
        • K-Means clustering
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">

        <div class="hero-title">
        🍔 Food Delivery Pattern Mining
        </div>

        <div class="hero-subtitle">
        Discovering meaningful patterns in food delivery operations
        using data mining techniques.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        This project analyzes food delivery operations to identify
        patterns related to delivery time, order timing, traffic,
        geographic distance, weather and delivery profiles.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    avg_time = df["Time_taken (min)"].mean()
    median_time = df["Time_taken (min)"].median()

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Total Orders",
            f"{len(df):,}"
        )

    with k2:
        st.metric(
            "Features",
            f"{len(df.columns)}"
        )

    with k3:
        st.metric(
            "Average Delivery Time",
            f"{avg_time:.2f} min"
        )

    with k4:
        st.metric(
            "Median Delivery Time",
            f"{median_time:.0f} min"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # ANALYSIS AREAS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">What This Project Analyzes</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            ⏱ Temporal Patterns
            </div>

            <div class="info-card-text">
            Analyze delivery performance across order hours,
            days of the week and weekends.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            📍 Geographic Patterns
            </div>

            <div class="info-card-text">
            Calculate delivery distance using the Haversine
            formula and examine its relationship with delivery time.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            🚦 Operational Patterns
            </div>

            <div class="info-card-text">
            Compare delivery performance across traffic,
            weather and vehicle conditions.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            📊 K-Means Clustering
            </div>

            <div class="info-card-text">
            Group deliveries into distinct operational
            profiles based on selected numerical features.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            🌦 Environmental Profiling
            </div>

            <div class="info-card-text">
            Examine how weather conditions are distributed
            across the discovered delivery profiles.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c6:
        st.markdown(
            """
            <div class="info-card">

            <div class="info-card-title">
            🛵 Vehicle Profiling
            </div>

            <div class="info-card-text">
            Compare vehicle-type distributions across
            the identified delivery clusters.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # KEY FINDINGS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Key Findings</div>',
        unsafe_allow_html=True
    )

    distance_df = get_distance_dataset(
        df_prepared,
        max_distance=50
    )

    correlation = get_distance_time_correlation(
        distance_df
    )

    traffic_data = get_traffic_pattern(df)

    if not traffic_data.empty:
        highest_traffic_row = traffic_data.loc[
            traffic_data["Average_Delivery_Time"].idxmax()
        ]

        highest_traffic = highest_traffic_row[
            "Road_traffic_density"
        ]

        highest_traffic_time = highest_traffic_row[
            "Average_Delivery_Time"
        ]
    else:
        highest_traffic = "N/A"
        highest_traffic_time = 0

    hourly_data = get_hourly_delivery_pattern(
        df_prepared
    )

    if not hourly_data.empty:
        highest_hour_row = hourly_data.loc[
            hourly_data["Average_Delivery_Time"].idxmax()
        ]

        highest_hour = int(
            highest_hour_row["Order_Hour"]
        )

        highest_hour_time = highest_hour_row[
            "Average_Delivery_Time"
        ]
    else:
        highest_hour = 0
        highest_hour_time = 0

    f1, f2 = st.columns(2)

    with f1:

        st.markdown(
            f"""
            <div class="finding-card">

            <div class="finding-title">
            📍 Delivery distance shows a positive relationship
            </div>

            <div class="finding-text">
            The Pearson correlation between delivery distance
            and delivery time is approximately
            <b>{correlation:.3f}</b>.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="finding-card">

            <div class="finding-title">
            🚦 Traffic conditions matter operationally
            </div>

            <div class="finding-text">
            <b>{highest_traffic}</b> traffic has the highest
            average delivery time in the operational analysis,
            at approximately <b>{highest_traffic_time:.2f} minutes</b>.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with f2:

        st.markdown(
            f"""
            <div class="finding-card">

            <div class="finding-title">
            🕐 Delivery time varies by order hour
            </div>

            <div class="finding-text">
            The highest observed average delivery time occurs
            around hour <b>{highest_hour}:00</b>, at approximately
            <b>{highest_hour_time:.2f} minutes</b>.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="finding-card">

            <div class="finding-title">
            📊 K-Means identifies delivery profiles
            </div>

            <div class="finding-text">
            Five clusters are used as a practical and interpretable
            solution to characterize different delivery profiles.
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # METHODOLOGY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Data Mining Workflow</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    methods = [
        ("01", "Preprocessing"),
        ("02", "Temporal Analysis"),
        ("03", "Geographic Analysis"),
        ("04", "Operational Analysis"),
        ("05", "K-Means Clustering")
    ]

    for column, (number, name) in zip(
        [m1, m2, m3, m4, m5],
        methods
    ):

        with column:

            st.markdown(
                f"""
                <div class="method-card">

                <div class="method-number">
                {number}
                </div>

                <div class="method-name">
                {name}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Summary of the Zomato delivery operations dataset,
        including data quality, structure and sample records.
        </div>
        """,
        unsafe_allow_html=True
    )

    overview = get_dataset_overview(df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            f"{overview['rows']:,}"
        )

    with c2:
        st.metric(
            "Columns",
            f"{overview['columns']}"
        )

    with c3:
        st.metric(
            "Duplicates",
            f"{overview['duplicates']:,}"
        )

    with c4:
        st.metric(
            "Missing Values",
            f"{int(df.isnull().sum().sum()):,}"
        )

    st.markdown("---")

    # Date range

    d1, d2 = st.columns(2)

    with d1:
        st.info(
            f"**Start Date:** "
            f"{overview['start_date'].strftime('%d %B %Y')}"
        )

    with d2:
        st.info(
            f"**End Date:** "
            f"{overview['end_date'].strftime('%d %B %Y')}"
        )

    st.markdown(
        '<div class="section-title">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        get_dataset_preview(df, 10),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Missing Value Analysis</div>',
        unsafe_allow_html=True
    )

    missing = get_missing_values(df)

    missing_df = (
        missing[missing > 0]
        .sort_values(ascending=False)
        .reset_index()
    )

    missing_df.columns = [
        "Column",
        "Missing Values"
    ]

    if missing_df.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        '<div class="section-title">Data Types</div>',
        unsafe_allow_html=True
    )

    dtype_df = (
        get_data_types(df)
        .astype(str)
        .reset_index()
    )

    dtype_df.columns = [
        "Column",
        "Data Type"
    ]

    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# GEOGRAPHIC ANALYSIS
# =========================================================

elif page == "Geographic Analysis":

    st.markdown(
        '<div class="section-title">Geographic Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Delivery distance is calculated using the Haversine
        formula and analyzed against delivery time.
        </div>
        """,
        unsafe_allow_html=True
    )

    distance_df = get_distance_dataset(
        df_prepared,
        max_distance=50
    )

    stats = get_distance_statistics(
        distance_df
    )

    correlation = get_distance_time_correlation(
        distance_df
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Valid Records",
            f"{stats['count']:,}"
        )

    with c2:
        st.metric(
            "Average Distance",
            f"{stats['mean']:.2f} km"
        )

    with c3:
        st.metric(
            "Median Distance",
            f"{stats['median']:.2f} km"
        )

    with c4:
        st.metric(
            "Distance-Time Correlation",
            f"{correlation:.3f}"
        )

    st.markdown("---")

    st.plotly_chart(
        create_distance_time_plot(distance_df),
        use_container_width=True
    )

    st.info(
        f"""
        **Interpretation:** The Pearson correlation between
        delivery distance and delivery time is approximately
        **{correlation:.3f}**, indicating a positive relationship
        in the analyzed records.
        """
    )

    st.markdown(
        '<div class="section-title">Distance Statistics</div>',
        unsafe_allow_html=True
    )

    distance_stats_df = pd.DataFrame(
        {
            "Metric": [
                "Number of Records",
                "Average Distance",
                "Median Distance",
                "Minimum Distance",
                "Maximum Distance"
            ],
            "Value": [
                f"{stats['count']:,}",
                f"{stats['mean']:.2f} km",
                f"{stats['median']:.2f} km",
                f"{stats['min']:.2f} km",
                f"{stats['max']:.2f} km"
            ]
        }
    )

    st.dataframe(
        distance_stats_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# TEMPORAL ANALYSIS
# =========================================================

elif page == "Temporal Analysis":

    st.markdown(
        '<div class="section-title">Temporal Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Explore how delivery performance changes across order
        hours, days of the week and weekend periods.
        </div>
        """,
        unsafe_allow_html=True
    )

    hourly_data = get_hourly_delivery_pattern(
        df_prepared
    )

    day_data = get_day_of_week_pattern(
        df_prepared
    )

    weekend_data = get_weekend_pattern(
        df_prepared
    )

    # -----------------------------------------------------
    # HOURLY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Order Hour Pattern</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        create_hourly_delivery_plot(df_prepared),
        use_container_width=True
    )

    st.dataframe(
        hourly_data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # DAY OF WEEK
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Day of Week Pattern</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(
        create_day_of_week_plot(df_prepared),
        use_container_width=True
    )

    st.dataframe(
        day_data,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # WEEKEND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Weekday vs Weekend</div>',
        unsafe_allow_html=True
    )

    wc1, wc2 = st.columns(2)

    with wc1:

        fig = px.bar(
            weekend_data,
            x="Day_Type",
            y="Average_Delivery_Time",
            title="Average Delivery Time",
            labels={
                "Day_Type": "Day Type",
                "Average_Delivery_Time":
                    "Average Delivery Time (minutes)"
            }
        )

        fig.update_layout(height=430)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with wc2:

        fig = px.bar(
            weekend_data,
            x="Day_Type",
            y="Order_Count",
            title="Number of Orders",
            labels={
                "Day_Type": "Day Type",
                "Order_Count": "Order Count"
            }
        )

        fig.update_layout(height=430)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.dataframe(
        weekend_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# OPERATIONAL ANALYSIS
# =========================================================

elif page == "Operational Analysis":

    st.markdown(
        '<div class="section-title">Operational Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        Analyze delivery performance across different
        road traffic conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

    traffic_data = get_traffic_pattern(df)

    # -----------------------------------------------------
    # TRAFFIC METRICS
    # -----------------------------------------------------

    if not traffic_data.empty:

        highest_time = traffic_data[
            "Average_Delivery_Time"
        ].max()

        highest_condition = traffic_data.loc[
            traffic_data[
                "Average_Delivery_Time"
            ].idxmax(),
            "Road_traffic_density"
        ]

        total_orders = traffic_data[
            "Order_Count"
        ].sum()

    else:

        highest_time = 0
        highest_condition = "N/A"
        total_orders = 0

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Traffic Records",
            f"{total_orders:,}"
        )

    with c2:
        st.metric(
            "Highest Average Time",
            f"{highest_time:.2f} min"
        )

    with c3:
        st.metric(
            "Condition",
            str(highest_condition)
        )

    st.markdown("---")

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    oc1, oc2 = st.columns(2)

    with oc1:

        fig = px.bar(
            traffic_data,
            x="Road_traffic_density",
            y="Average_Delivery_Time",
            title="Average Delivery Time by Traffic Density",
            labels={
                "Road_traffic_density":
                    "Traffic Density",
                "Average_Delivery_Time":
                    "Average Delivery Time (minutes)"
            }
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with oc2:

        fig = px.bar(
            traffic_data,
            x="Road_traffic_density",
            y="Order_Count",
            title="Orders by Traffic Density",
            labels={
                "Road_traffic_density":
                    "Traffic Density",
                "Order_Count":
                    "Order Count"
            }
        )

        fig.update_layout(height=450)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.dataframe(
        traffic_data,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        **Interpretation:** Traffic density is associated with
        different delivery-time patterns in the dataset.
        These observations describe associations in the data
        and should not be interpreted as causal relationships.
        """
    )


# =========================================================
# K-MEANS CLUSTERING
# =========================================================

elif page == "K-Means Clustering":

    st.markdown(
        '<div class="section-title">K-Means Clustering</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-description">
        K-Means clustering is used to identify distinct delivery
        profiles using delivery distance, delivery time,
        delivery-person rating and multiple-delivery workload.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # PREPARE CLUSTERING DATA
    # -----------------------------------------------------

    df_cluster_prepared, cluster_df, features = (
        prepare_cluster_data(df)
    )

    # -----------------------------------------------------
    # DATASET METRICS
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Original Records",
            f"{len(df):,}"
        )

    with c2:
        st.metric(
            "Valid Distance Records",
            f"{len(df_cluster_prepared):,}"
        )

    with c3:
        st.metric(
            "Clustering Records",
            f"{len(cluster_df):,}"
        )

    with c4:
        st.metric(
            "Excluded",
            f"{len(df_cluster_prepared) - len(cluster_df):,}"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Features Used for Clustering</div>',
        unsafe_allow_html=True
    )

    feature_cols = st.columns(len(features))

    feature_names = {
        "Delivery_Distance_km": "Delivery Distance",
        "Time_taken (min)": "Delivery Time",
        "Delivery_person_Ratings": "Delivery Rating",
        "multiple_deliveries": "Multiple Deliveries"
    }

    for column, feature in zip(
        feature_cols,
        features
    ):

        with column:

            st.markdown(
                f"""
                <div class="info-card">

                <div class="info-card-title">
                {feature_names.get(feature, feature)}
                </div>

                <div class="info-card-text">
                {feature}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------------------------------
    # FEATURE STATISTICS
    # -----------------------------------------------------

    with st.expander(
        "View clustering feature statistics"
    ):

        st.dataframe(
            cluster_df[features]
            .describe()
            .round(2),
            use_container_width=True
        )

    # -----------------------------------------------------
    # SCALE
    # -----------------------------------------------------

    scaled_data, scaler = scale_features(
        cluster_df,
        features
    )

    # -----------------------------------------------------
    # ELBOW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">1. Elbow Method</div>',
        unsafe_allow_html=True
    )

    elbow_data = calculate_elbow_values(
        scaled_data,
        min_k=2,
        max_k=8
    )

    st.plotly_chart(
        create_elbow_plot(elbow_data),
        use_container_width=True
    )

    st.dataframe(
        elbow_data.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        **Cluster selection:** The elbow trend is considered
        together with silhouette analysis and cluster
        interpretability. K = 5 is selected as a practical
        solution for this project.
        """
    )

    # -----------------------------------------------------
    # SILHOUETTE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">2. Silhouette Analysis</div>',
        unsafe_allow_html=True
    )

    silhouette_data = calculate_silhouette_values(
        scaled_data,
        min_k=2,
        max_k=8,
        sample_size=10000
    )

    st.plotly_chart(
        create_silhouette_plot(
            silhouette_data
        ),
        use_container_width=True
    )

    st.dataframe(
        silhouette_data.round(3),
        use_container_width=True,
        hide_index=True
    )

    best_row = silhouette_data.loc[
        silhouette_data["Silhouette_Score"].idxmax()
    ]

    best_k = int(best_row["K"])
    best_score = best_row["Silhouette_Score"]

    st.info(
        f"""
        The highest sampled silhouette score is
        **{best_score:.3f} at K = {best_k}**.
        K = 5 is retained as the final solution based on
        the elbow trend and interpretability of the resulting
        delivery profiles.
        """
    )

    # -----------------------------------------------------
    # FINAL K-MEANS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">3. Final K-Means Profiles</div>',
        unsafe_allow_html=True
    )

    clustered_df, model, final_scaler = perform_kmeans(
        cluster_df,
        features,
        n_clusters=5
    )

    profile = get_cluster_profile(
        clustered_df
    )

    # -----------------------------------------------------
    # PROFILE TABLE
    # -----------------------------------------------------

    profile_display = profile.rename(
        columns={
            "Cluster": "Cluster",
            "Delivery_Distance_km": "Distance (km)",
            "Delivery_Time": "Delivery Time (min)",
            "Delivery_person_Ratings": "Rating",
            "Multiple_Deliveries": "Multiple Deliveries",
            "Count": "Records",
            "Percentage": "Share (%)"
        }
    )

    st.dataframe(
        profile_display,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # PROFILE CHARTS
    # -----------------------------------------------------

    pc1, pc2 = st.columns(2)

    with pc1:

        st.plotly_chart(
            create_cluster_time_plot(
                profile
            ),
            use_container_width=True
        )

    with pc2:

        st.plotly_chart(
            create_cluster_size_plot(
                profile
            ),
            use_container_width=True
        )

    # -----------------------------------------------------
    # PROFILE INTERPRETATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Delivery Profile Interpretation</div>',
        unsafe_allow_html=True
    )

    for _, row in profile.iterrows():

        cluster = int(row["Cluster"])
        distance = row["Delivery_Distance_km"]
        delivery_time = row["Delivery_Time"]
        rating = row["Delivery_person_Ratings"]
        multiple = row["Multiple_Deliveries"]
        count = int(row["Count"])
        percentage = row["Percentage"]

        st.markdown(
            f"""
            <div class="finding-card">

            <div class="finding-title">
            Cluster {cluster}
            </div>

            <div class="finding-text">
            {count:,} records ({percentage:.2f}% of clustering data)
            • Distance: {distance:.2f} km
            • Delivery time: {delivery_time:.2f} min
            • Rating: {rating:.2f}
            • Multiple deliveries: {multiple:.2f}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # TRAFFIC PROFILE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">4. Traffic Distribution Across Clusters</div>',
        unsafe_allow_html=True
    )

    traffic_profile = get_traffic_profile(
        df,
        clustered_df
    )

    st.plotly_chart(
        create_traffic_heatmap(
            traffic_profile
        ),
        use_container_width=True
    )

    with st.expander(
        "View traffic profile data"
    ):

        st.dataframe(
            traffic_profile,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # WEATHER PROFILE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">5. Weather Distribution Across Clusters</div>',
        unsafe_allow_html=True
    )

    weather_profile = get_weather_profile(
        df,
        clustered_df
    )

    st.plotly_chart(
        create_weather_heatmap(
            weather_profile
        ),
        use_container_width=True
    )

    with st.expander(
        "View weather profile data"
    ):

        st.dataframe(
            weather_profile,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # VEHICLE PROFILE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">6. Vehicle Type Distribution Across Clusters</div>',
        unsafe_allow_html=True
    )

    vehicle_profile = get_vehicle_profile(
        df,
        clustered_df
    )

    st.plotly_chart(
        create_vehicle_heatmap(
            vehicle_profile
        ),
        use_container_width=True
    )

    with st.expander(
        "View vehicle profile data"
    ):

        st.dataframe(
            vehicle_profile,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # CLUSTERING NOTE
    # -----------------------------------------------------

    st.warning(
        """
        **Interpretation note:** K-Means identifies patterns
        in the selected numerical features; it does not establish
        causation. Traffic, weather and vehicle distributions are
        shown as post-clustering profiles. Since delivery time is
        itself one of the clustering features, the clusters are
        interpreted as **delivery profiles**, not independent
        causal groups.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>Food Delivery Pattern Mining</b><br>

    Data Mining Project • Temporal Analysis • Geographic Analysis
    • Operational Analysis • K-Means Clustering

    </div>
    """,
    unsafe_allow_html=True
)