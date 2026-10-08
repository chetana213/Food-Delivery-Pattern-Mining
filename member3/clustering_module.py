import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# 1. Haversine Distance
# ---------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


# ---------------------------------------------------------
# 2. Calculate Delivery Distance
# ---------------------------------------------------------

def calculate_delivery_distance(df):

    df = df.copy()

    df["Delivery_Distance_km"] = haversine(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"]
    )

    return df


# ---------------------------------------------------------
# 3. Prepare Clustering Data
# ---------------------------------------------------------

def prepare_clustering_data(df, max_distance=50):

    df = calculate_delivery_distance(df)

    # Remove unrealistic geographic distances
    df = df[
        df["Delivery_Distance_km"] <= max_distance
    ].copy()

    cluster_features = [
        "Delivery_Distance_km",
        "Time_taken (min)",
        "Delivery_person_Ratings",
        "multiple_deliveries"
    ]

    # Remove missing values only from clustering features
    cluster_df = df[
        cluster_features
    ].dropna().copy()

    return df, cluster_df, cluster_features


# ---------------------------------------------------------
# 4. Standardize Features
# ---------------------------------------------------------

def scale_features(cluster_df, features):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        cluster_df[features]
    )

    return scaled_data, scaler


# ---------------------------------------------------------
# 5. Elbow Method
# ---------------------------------------------------------

def calculate_elbow_values(
    scaled_data,
    min_k=2,
    max_k=8
):

    results = []

    for k in range(min_k, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(scaled_data)

        results.append({
            "K": k,
            "Inertia": model.inertia_
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 6. Elbow Plot
# ---------------------------------------------------------

def create_elbow_plot(elbow_data):

    fig = px.line(
        elbow_data,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method for Selecting K",
        labels={
            "K": "Number of Clusters",
            "Inertia": "Inertia"
        }
    )

    fig.update_layout(height=500)

    return fig


# ---------------------------------------------------------
# 7. Silhouette Analysis - FAST VERSION
# ---------------------------------------------------------

def calculate_silhouette_values(
    scaled_data,
    min_k=2,
    max_k=8,
    sample_size=10000
):

    results = []

    # Use a fixed sample for faster calculation
    if len(scaled_data) > sample_size:

        rng = np.random.RandomState(42)

        sample_indices = rng.choice(
            len(scaled_data),
            size=sample_size,
            replace=False
        )

        silhouette_data = scaled_data[
            sample_indices
        ]

    else:

        silhouette_data = scaled_data

    print(
        f"\nSilhouette analysis using "
        f"{len(silhouette_data):,} records..."
    )

    for k in range(min_k, max_k + 1):

        print(f"Calculating silhouette score for K={k}...")

        # Fit K-Means on the sampled data
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(
            silhouette_data
        )

        score = silhouette_score(
            silhouette_data,
            labels
        )

        results.append({
            "K": k,
            "Silhouette_Score": score
        })

        print(
            f"K={k} -> Silhouette Score: "
            f"{score:.4f}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# 8. Silhouette Plot
# ---------------------------------------------------------

def create_silhouette_plot(
    silhouette_data
):

    fig = px.line(
        silhouette_data,
        x="K",
        y="Silhouette_Score",
        markers=True,
        title="Silhouette Score for Different K Values",
        labels={
            "K": "Number of Clusters",
            "Silhouette_Score": "Silhouette Score"
        }
    )

    fig.update_layout(height=500)

    return fig


# ---------------------------------------------------------
# 9. Final K-Means Clustering
# ---------------------------------------------------------

def perform_kmeans(
    cluster_df,
    features,
    n_clusters=5
):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        cluster_df[features]
    )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        scaled_data
    )

    result = cluster_df.copy()

    result["Cluster"] = labels

    return result, model, scaler


# ---------------------------------------------------------
# 10. Cluster Profile
# ---------------------------------------------------------

def get_cluster_profile(
    clustered_df
):

    profile = (
        clustered_df
        .groupby("Cluster")
        .agg(
            Delivery_Distance_km=(
                "Delivery_Distance_km",
                "mean"
            ),

            Delivery_Time=(
                "Time_taken (min)",
                "mean"
            ),

            Delivery_person_Ratings=(
                "Delivery_person_Ratings",
                "mean"
            ),

            Multiple_Deliveries=(
                "multiple_deliveries",
                "mean"
            ),

            Count=(
                "Cluster",
                "size"
            )
        )
        .reset_index()
    )

    total = len(clustered_df)

    profile["Percentage"] = (
        profile["Count"] / total * 100
    )

    return profile.round(2)


# ---------------------------------------------------------
# 11. Average Delivery Time by Cluster
# ---------------------------------------------------------

def create_cluster_time_plot(profile):

    fig = px.bar(
        profile,
        x="Cluster",
        y="Delivery_Time",
        title="Average Delivery Time by Cluster",
        labels={
            "Cluster": "Cluster",
            "Delivery_Time":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(height=500)

    return fig


# ---------------------------------------------------------
# 12. Cluster Size Plot
# ---------------------------------------------------------

def create_cluster_size_plot(profile):

    fig = px.bar(
        profile,
        x="Cluster",
        y="Count",
        title="Number of Records in Each Cluster",
        labels={
            "Cluster": "Cluster",
            "Count": "Number of Records"
        }
    )

    fig.update_layout(height=500)

    return fig


# ---------------------------------------------------------
# 13. Category Profile
# ---------------------------------------------------------

def get_cluster_category_pattern(
    full_df,
    clustered_df,
    column
):

    category_data = (
        full_df[
            ["ID", column]
        ]
        .dropna()
    )

    common_indices = (
        category_data.index
        .intersection(clustered_df.index)
    )

    category_data = category_data.loc[
        common_indices
    ].copy()

    category_data["Cluster"] = (
        clustered_df
        .loc[common_indices, "Cluster"]
    )

    result = (
        category_data
        .groupby(
            ["Cluster", column]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    result["Percentage"] = (
        result
        .groupby("Cluster")["Count"]
        .transform(
            lambda x:
                x / x.sum() * 100
        )
    )

    result["Percentage"] = (
        result["Percentage"]
        .round(2)
    )

    return result


# ---------------------------------------------------------
# 14. Traffic Profile
# ---------------------------------------------------------

def get_traffic_profile(
    full_df,
    clustered_df
):

    return get_cluster_category_pattern(
        full_df,
        clustered_df,
        "Road_traffic_density"
    )


# ---------------------------------------------------------
# 15. Weather Profile
# ---------------------------------------------------------

def get_weather_profile(
    full_df,
    clustered_df
):

    return get_cluster_category_pattern(
        full_df,
        clustered_df,
        "Weather_conditions"
    )


# ---------------------------------------------------------
# 16. Vehicle Profile
# ---------------------------------------------------------

def get_vehicle_profile(
    full_df,
    clustered_df
):

    return get_cluster_category_pattern(
        full_df,
        clustered_df,
        "Type_of_vehicle"
    )


# ---------------------------------------------------------
# 17. Traffic Heatmap
# ---------------------------------------------------------

def create_traffic_heatmap(
    traffic_profile
):

    pivot = traffic_profile.pivot(
        index="Cluster",
        columns="Road_traffic_density",
        values="Percentage"
    )

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        title="Traffic Distribution Across Clusters",
        labels={
            "x": "Traffic Density",
            "y": "Cluster",
            "color": "Percentage"
        }
    )

    fig.update_layout(height=500)

    return fig


# ---------------------------------------------------------
# 18. Weather Heatmap
# ---------------------------------------------------------

def create_weather_heatmap(
    weather_profile
):

    pivot = weather_profile.pivot(
        index="Cluster",
        columns="Weather_conditions",
        values="Percentage"
    )

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        title="Weather Distribution Across Clusters",
        labels={
            "x": "Weather Condition",
            "y": "Cluster",
            "color": "Percentage"
        }
    )

    fig.update_layout(height=550)

    return fig


# ---------------------------------------------------------
# 19. Vehicle Heatmap
# ---------------------------------------------------------

def create_vehicle_heatmap(
    vehicle_profile
):

    pivot = vehicle_profile.pivot(
        index="Cluster",
        columns="Type_of_vehicle",
        values="Percentage"
    )

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        title="Vehicle Type Distribution Across Clusters",
        labels={
            "x": "Vehicle Type",
            "y": "Cluster",
            "color": "Percentage"
        }
    )

    fig.update_layout(height=500)

    return fig