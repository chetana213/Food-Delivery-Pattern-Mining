import pandas as pd
import plotly.express as px


# ============================================================
# PREPARE TEMPORAL FEATURES
# ============================================================

def prepare_temporal_features(df):
    """
    Create temporal features from order date and order time.
    """

    df = df.copy()

    # Convert order date
    df["Order_Date_temp"] = pd.to_datetime(
        df["Order_Date"],
        format="%d-%m-%Y"
    )

    # Extract order hour
    df["Order_Hour"] = pd.to_datetime(
        df["Time_Orderd"],
        format="%H:%M",
        errors="coerce"
    ).dt.hour

    # Extract day of week
    df["Day_of_Week"] = (
        df["Order_Date_temp"]
        .dt.day_name()
    )

    # Identify weekend
    df["Weekend"] = (
        df["Order_Date_temp"]
        .dt.dayofweek >= 5
    )

    # Extract month
    df["Month"] = (
        df["Order_Date_temp"]
        .dt.month
    )

    return df


# ============================================================
# ORDER HOUR ANALYSIS
# ============================================================

def get_hourly_delivery_pattern(df):
    """
    Calculate order count and average delivery time
    for each order hour.
    """

    result = (
        df.dropna(subset=["Order_Hour"])
        .groupby("Order_Hour")
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
# DAY OF WEEK ANALYSIS
# ============================================================

def get_day_of_week_pattern(df):
    """
    Calculate order count and average delivery time
    for each day of the week.
    """

    result = (
        df.groupby("Day_of_Week")
        .agg(
            Order_Count=("ID", "count"),
            Average_Delivery_Time=(
                "Time_taken (min)",
                "mean"
            )
        )
        .reset_index()
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    result["Day_of_Week"] = pd.Categorical(
        result["Day_of_Week"],
        categories=day_order,
        ordered=True
    )

    result = result.sort_values(
        "Day_of_Week"
    )

    result["Average_Delivery_Time"] = (
        result["Average_Delivery_Time"]
        .round(2)
    )

    return result


# ============================================================
# WEEKEND VS WEEKDAY
# ============================================================

def get_weekend_pattern(df):
    """
    Compare delivery patterns between weekdays
    and weekends.
    """

    result = (
        df.groupby("Weekend")
        .agg(
            Order_Count=("ID", "count"),
            Average_Delivery_Time=(
                "Time_taken (min)",
                "mean"
            )
        )
        .reset_index()
    )

    result["Day_Type"] = result["Weekend"].map({
        False: "Weekday",
        True: "Weekend"
    })

    result["Average_Delivery_Time"] = (
        result["Average_Delivery_Time"]
        .round(2)
    )

    return result[
        [
            "Day_Type",
            "Order_Count",
            "Average_Delivery_Time"
        ]
    ]


# ============================================================
# HOURLY DELIVERY TIME PLOT
# ============================================================

def create_hourly_delivery_plot(df):
    """
    Create an interactive line chart showing
    average delivery time by order hour.
    """

    data = get_hourly_delivery_pattern(df)

    fig = px.line(
        data,
        x="Order_Hour",
        y="Average_Delivery_Time",
        markers=True,
        title="Order Hour vs Average Delivery Time",
        labels={
            "Order_Hour": "Order Hour",
            "Average_Delivery_Time":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(
        height=500
    )

    return fig


# ============================================================
# DAY OF WEEK PLOT
# ============================================================

def create_day_of_week_plot(df):
    """
    Create an interactive bar chart showing
    average delivery time by day of week.
    """

    data = get_day_of_week_pattern(df)

    fig = px.bar(
        data,
        x="Day_of_Week",
        y="Average_Delivery_Time",
        title="Day of Week vs Average Delivery Time",
        labels={
            "Day_of_Week": "Day of Week",
            "Average_Delivery_Time":
                "Average Delivery Time (minutes)"
        }
    )

    fig.update_layout(
        height=500
    )

    return fig