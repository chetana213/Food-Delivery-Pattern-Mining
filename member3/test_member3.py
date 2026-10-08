import pandas as pd

from clustering_module import (
    prepare_clustering_data,
    scale_features,
    calculate_elbow_values,
    calculate_silhouette_values,
    perform_kmeans,
    get_cluster_profile,
    get_traffic_profile,
    get_weather_profile,
    get_vehicle_profile
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("\nDATASET")
print("----------------")

df = pd.read_csv(
    "Zomato Dataset.csv"
)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ---------------------------------------------------------
# PREPARE CLUSTERING DATA
# ---------------------------------------------------------

print("\nCLUSTERING DATA")
print("----------------")

df_prepared, cluster_df, features = (
    prepare_clustering_data(df)
)

print(
    "Valid geographic records:",
    len(df_prepared)
)

print(
    "Records used for clustering:",
    len(cluster_df)
)

print(
    "Records excluded from clustering:",
    len(df_prepared) - len(cluster_df)
)


print("\nCLUSTERING FEATURES")

for feature in features:
    print("-", feature)


# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

scaled_data, scaler = scale_features(
    cluster_df,
    features
)

print("\nFEATURE SCALING")
print("----------------")

print(
    "StandardScaler applied successfully."
)


# ---------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------

print("\nELBOW METHOD")
print("----------------")

elbow_data = calculate_elbow_values(
    scaled_data,
    min_k=2,
    max_k=8
)

print(elbow_data)


# ---------------------------------------------------------
# SILHOUETTE ANALYSIS
# ---------------------------------------------------------

print("\nSILHOUETTE ANALYSIS")
print("-------------------")

silhouette_data = calculate_silhouette_values(
    scaled_data,
    min_k=2,
    max_k=8
)

print(silhouette_data)


# ---------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------

print("\nK-MEANS CLUSTERING")
print("------------------")

clustered_df, model, scaler = perform_kmeans(
    cluster_df,
    features,
    n_clusters=5
)

print(
    "Number of clusters:",
    5
)

print("\nCluster counts:")

print(
    clustered_df["Cluster"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

print("\nCLUSTER PROFILE")
print("----------------")

profile = get_cluster_profile(
    clustered_df
)

print(profile)


# ---------------------------------------------------------
# TRAFFIC PROFILE
# ---------------------------------------------------------

print("\nTRAFFIC PROFILE")
print("----------------")

traffic_profile = get_traffic_profile(
    df,
    clustered_df
)

print(traffic_profile)


# ---------------------------------------------------------
# WEATHER PROFILE
# ---------------------------------------------------------

print("\nWEATHER PROFILE")
print("----------------")

weather_profile = get_weather_profile(
    df,
    clustered_df
)

print(weather_profile)


# ---------------------------------------------------------
# VEHICLE PROFILE
# ---------------------------------------------------------

print("\nVEHICLE PROFILE")
print("----------------")

vehicle_profile = get_vehicle_profile(
    df,
    clustered_df
)

print(vehicle_profile)


# ---------------------------------------------------------
# COMPLETION
# ---------------------------------------------------------

print(
    "\nMEMBER 3 TEST COMPLETED SUCCESSFULLY"
)