import streamlit as st
import pandas as pd

from clustering_module import (
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


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Food Delivery Pattern Mining - Member 3",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("Food Delivery Pattern Mining")

st.subheader(
    "Member 3 — K-Means Clustering Analysis"
)

st.write(
    """
    This module uses K-Means clustering to discover
    different delivery profiles based on delivery
    distance, delivery time, delivery-person rating,
    and multiple deliveries.
    """
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    return pd.read_csv(
        "Zomato Dataset.csv"
    )


df = load_data()


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

df_prepared, cluster_df, features = (
    prepare_clustering_data(df)
)


# ---------------------------------------------------------
# DATASET SUMMARY
# ---------------------------------------------------------

st.header("1. Clustering Dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Records",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Valid Distance Records",
        f"{len(df_prepared):,}"
    )

with col3:
    st.metric(
        "Records Used",
        f"{len(cluster_df):,}"
    )

with col4:
    st.metric(
        "Excluded",
        f"{len(df_prepared) - len(cluster_df):,}"
    )


st.write(
    "**Features used for clustering:**"
)

st.write(
    ", ".join(features)
)


# ---------------------------------------------------------
# FEATURE STATISTICS
# ---------------------------------------------------------

st.subheader(
    "Clustering Feature Statistics"
)

st.dataframe(
    cluster_df[features]
    .describe()
    .round(2),
    use_container_width=True
)


# ---------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------

scaled_data, scaler = scale_features(
    cluster_df,
    features
)


# ---------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------

st.header("2. Elbow Method")

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
    use_container_width=True
)

st.info(
    """
    The Elbow Method shows how within-cluster variation
    decreases as the number of clusters increases.
    K = 5 is selected as a practical and interpretable
    clustering solution for this project.
    """
)


# ---------------------------------------------------------
# SILHOUETTE
# ---------------------------------------------------------

st.header("3. Silhouette Analysis")

silhouette_data = calculate_silhouette_values(
    scaled_data,
    min_k=2,
    max_k=8
)

st.plotly_chart(
    create_silhouette_plot(
        silhouette_data
    ),
    use_container_width=True
)

st.dataframe(
    silhouette_data.round(3),
    use_container_width=True
)

st.info(
    """
    The silhouette scores indicate moderate cluster
    separation. Therefore, the clusters are interpreted
    as useful delivery profiles rather than absolute
    categories.
    """
)


# ---------------------------------------------------------
# FINAL K-MEANS
# ---------------------------------------------------------

st.header("4. K-Means Clustering")

clustered_df, model, scaler = perform_kmeans(
    cluster_df,
    features,
    n_clusters=5
)


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

profile = get_cluster_profile(
    clustered_df
)

st.subheader(
    "Cluster Profiles"
)

st.dataframe(
    profile,
    use_container_width=True
)


# ---------------------------------------------------------
# CLUSTER CHARTS
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        create_cluster_time_plot(
            profile
        ),
        use_container_width=True
    )

with col2:

    st.plotly_chart(
        create_cluster_size_plot(
            profile
        ),
        use_container_width=True
    )


# ---------------------------------------------------------
# CLUSTER INTERPRETATION
# ---------------------------------------------------------

st.subheader(
    "Cluster Interpretation"
)

for _, row in profile.iterrows():

    cluster = int(row["Cluster"])

    st.write(
        f"""
        **Cluster {cluster}:**
        {int(row["Count"]):,} records
        ({row["Percentage"]:.2f}%).

        Average distance:
        {row["Delivery_Distance_km"]:.2f} km

        Average delivery time:
        {row["Delivery_Time"]:.2f} minutes

        Average delivery-person rating:
        {row["Delivery_person_Ratings"]:.2f}

        Average multiple deliveries:
        {row["Multiple_Deliveries"]:.2f}
        """
    )


# ---------------------------------------------------------
# TRAFFIC PROFILING
# ---------------------------------------------------------

st.header(
    "5. Traffic Distribution Across Clusters"
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

st.dataframe(
    traffic_profile,
    use_container_width=True
)


# ---------------------------------------------------------
# WEATHER PROFILING
# ---------------------------------------------------------

st.header(
    "6. Weather Distribution Across Clusters"
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

st.dataframe(
    weather_profile,
    use_container_width=True
)


# ---------------------------------------------------------
# VEHICLE PROFILING
# ---------------------------------------------------------

st.header(
    "7. Vehicle Type Distribution Across Clusters"
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

st.dataframe(
    vehicle_profile,
    use_container_width=True
)


# ---------------------------------------------------------
# KEY FINDINGS
# ---------------------------------------------------------

st.header(
    "8. Key Findings"
)

st.success(
    """
    Five delivery profiles were identified using
    K-Means clustering.
    """
)

st.info(
    """
    Cluster 2 has the shortest average delivery
    distance and the lowest average delivery time,
    making it a relatively fast delivery profile.
    """
)

st.info(
    """
    Cluster 4 has the highest average delivery time,
    approximately 41.80 minutes.
    """
)

st.info(
    """
    Cluster 0 has the longest average delivery distance,
    approximately 16.45 km, but its average delivery
    time is only about 24.26 minutes. This shows that
    distance alone does not explain all delivery-time
    variation.
    """
)

st.info(
    """
    Cluster 4 has a high proportion of traffic-jam
    conditions, helping characterize it as a
    high-delay delivery profile.
    """
)


# ---------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------

with st.expander(
    "K-Means Methodology"
):

    st.write(
        """
        1. Calculate restaurant-to-delivery distance
           using the Haversine formula.

        2. Exclude records with distance above 50 km.

        3. Remove records with missing clustering
           feature values.

        4. Select four clustering features:
           distance, delivery time, rating and
           multiple deliveries.

        5. Standardize the features using
           StandardScaler.

        6. Evaluate K values using the Elbow Method
           and Silhouette Score.

        7. Select K = 5 for the final clustering model.

        8. Profile the clusters using traffic,
           weather and vehicle type.
        """
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Food Delivery Pattern Mining | Member 3 — K-Means Clustering"
)