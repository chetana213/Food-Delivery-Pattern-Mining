import pandas as pd
import plotly.express as px


# ============================================================
# TRAFFIC ANALYSIS
# ============================================================

def get_traffic_pattern(df):
    """
    Calculate order count and average delivery time
    for each traffic category.
    """

    result = (
        df.dropna(
            subset=["Road_traffic_density"]
        )
        .groupby("Road_traffic_density")
        .agg(
            Order_Count=("ID", "count"),
            Average_Delivery_Time=(
                "Time_taken (min)",
                "mean"
            )
        )
        .reset_index()
    )

    result["Average_Delivery_Time"] = (
        result["Average_Delivery_Time"]
        .round(2)
    )

    return result


# ============================================================
# WEATHER ANALYSIS
# ============================================================

def get_weather_pattern(df):
    """
    Calculate order count and average delivery time
    for each weather condition.
    """

    result = (
        df.dropna(
            subset=["Weather_conditions"]
        )
        .groupby("Weather_conditions")
        .agg(
            Order_Count=("ID", "count"),
            Average_Delivery_Time=(
                "Time_taken (min)",
                "mean"
            )
        )
        .reset_index()
    )

    result["Average_Delivery_Time"] = (
        result["Average_Delivery_Time"]
        .round(2)
    )

    return result


# ============================================================
# TRAFFIC × WEATHER ANALYSIS
# ============================================================

def get_traffic_weather_pattern(df):
    """
    Analyze delivery time across combinations
    of traffic density and weather conditions.
    """

    result = (
        df.dropna(
            subset=[
                "Road_traffic_density",
                "Weather_conditions"
            ]
        )
        .groupby(
            [
                "Road_traffic_density",
                "Weather_conditions"
            ]
        )
        .agg(
            Order_Count=("ID", "count"),
            Average_Delivery_Time=(
                "Time_taken (min)",
                "mean"
            )
        )
        .reset_index()
    )

    result["Average_Delivery_Time"] = (
        result["Average_Delivery_Time"]
        .round(2)
    )

    return result


# ============================================================
# TRAFFIC PLOT
# ============================================================

def create_traffic_plot(df):
    """
    Create an interactive bar chart showing
    traffic density versus delivery time.
    """

    data = get_traffic_pattern(df)

    fig = px.bar(
        data,
        x="Road_traffic_density",
        y="Average_Delivery_Time",
        title="Traffic Density vs Average Delivery Time",
        labels={
            "Road_traffic_density":
                "Traffic Density",
            "Average_Delivery_Time":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(
        height=500
    )

    return fig


# ============================================================
# WEATHER PLOT
# ============================================================

def create_weather_plot(df):
    """
    Create an interactive bar chart showing
    weather conditions versus delivery time.
    """

    data = get_weather_pattern(df)

    fig = px.bar(
        data,
        x="Weather_conditions",
        y="Average_Delivery_Time",
        title="Weather Conditions vs Average Delivery Time",
        labels={
            "Weather_conditions":
                "Weather Condition",
            "Average_Delivery_Time":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(
        height=500
    )

    return fig


# ============================================================
# TRAFFIC × WEATHER HEATMAP
# ============================================================

def create_traffic_weather_heatmap(df):
    """
    Create a heatmap showing average delivery time
    for combinations of traffic and weather.
    """

    data = get_traffic_weather_pattern(df)

    pivot_table = data.pivot(
        index="Road_traffic_density",
        columns="Weather_conditions",
        values="Average_Delivery_Time"
    )

    fig = px.imshow(
        pivot_table,
        text_auto=".1f",
        aspect="auto",
        title="Traffic Density × Weather Conditions",
        labels={
            "x": "Weather Condition",
            "y": "Traffic Density",
            "color":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(
        height=550
    )

    return fig