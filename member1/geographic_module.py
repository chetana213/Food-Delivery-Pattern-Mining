import numpy as np
import plotly.express as px


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate geographic distance between two coordinates
    using the Haversine formula.
    """

    R = 6371  # Earth radius in kilometres

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


def calculate_delivery_distance(df):
    """Calculate restaurant-to-delivery distance."""

    df = df.copy()

    df["Delivery_Distance_km"] = haversine(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"]
    )

    return df


def get_distance_dataset(df, max_distance=50):
    """
    Keep records with delivery distance within
    the specified valid range.
    """

    df_distance = df[
        df["Delivery_Distance_km"] <= max_distance
    ].copy()

    return df_distance


def get_distance_statistics(df_distance):
    """Return important distance statistics."""

    distance = df_distance["Delivery_Distance_km"]

    return {
        "count": int(distance.count()),
        "mean": float(distance.mean()),
        "median": float(distance.median()),
        "min": float(distance.min()),
        "max": float(distance.max())
    }


def get_distance_time_correlation(df_distance):
    """Calculate Pearson correlation between distance and delivery time."""

    correlation = df_distance[
        ["Delivery_Distance_km", "Time_taken (min)"]
    ].corr().iloc[0, 1]

    return float(correlation)


def create_distance_time_plot(df_distance):
    """
    Create an interactive scatter plot showing
    delivery distance versus delivery time.
    """

    correlation = get_distance_time_correlation(df_distance)

    fig = px.scatter(
        df_distance,
        x="Delivery_Distance_km",
        y="Time_taken (min)",
        title=f"Delivery Distance vs Delivery Time (r = {correlation:.3f})",
        labels={
            "Delivery_Distance_km": "Delivery Distance (km)",
            "Time_taken (min)": "Delivery Time (minutes)"
        },
        opacity=0.20
    )

    fig.update_traces(
        marker=dict(size=5)
    )

    fig.update_layout(
        xaxis_title="Delivery Distance (km)",
        yaxis_title="Delivery Time (minutes)",
        height=550
    )

    return fig