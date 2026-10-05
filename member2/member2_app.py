import streamlit as st
import pandas as pd

from temporal_module import (
    prepare_temporal_features,
    get_hourly_delivery_pattern,
    get_day_of_week_pattern,
    get_weekend_pattern,
    create_hourly_delivery_plot,
    create_day_of_week_plot
)

from operational_module import (
    get_traffic_pattern,
    get_weather_pattern,
    get_traffic_weather_pattern,
    create_traffic_plot,
    create_weather_plot,
    create_traffic_weather_heatmap
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Food Delivery Pattern Mining",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Food Delivery Pattern Mining"
)

st.markdown(
    """
    ### Temporal & Operational Pattern Analysis

    This section analyzes delivery-time patterns based on
    order timing, traffic conditions, and weather conditions.
    """
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "Zomato Dataset.csv"
    )


df = load_data()


# ============================================================
# PREPARE TEMPORAL FEATURES
# ============================================================

df = prepare_temporal_features(df)


# ============================================================
# TEMPORAL ANALYSIS
# ============================================================

st.header(
    "1. Temporal Analysis"
)


# ============================================================
# ORDER HOUR ANALYSIS
# ============================================================

st.subheader(
    "Order Hour vs Delivery Time"
)

st.plotly_chart(
    create_hourly_delivery_plot(df),
    use_container_width=True
)

hourly_data = (
    get_hourly_delivery_pattern(df)
)

st.dataframe(
    hourly_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# HOURLY FINDING
# ============================================================

st.info(
    """
    **Finding:** Delivery times are highest during the
    evening period around 19:00–21:00 and lowest during
    the morning period around 8:00–10:00.
    """
)


# ============================================================
# DAY OF WEEK
# ============================================================

st.subheader(
    "Day of Week vs Delivery Time"
)

st.plotly_chart(
    create_day_of_week_plot(df),
    use_container_width=True
)

day_data = (
    get_day_of_week_pattern(df)
)

st.dataframe(
    day_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# WEEKEND VS WEEKDAY
# ============================================================

st.subheader(
    "Weekend vs Weekday"
)

weekend_data = (
    get_weekend_pattern(df)
)

st.dataframe(
    weekend_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# OPERATIONAL ANALYSIS
# ============================================================

st.divider()

st.header(
    "2. Operational Analysis"
)


# ============================================================
# TRAFFIC ANALYSIS
# ============================================================

st.subheader(
    "Traffic Density vs Delivery Time"
)

st.plotly_chart(
    create_traffic_plot(df),
    use_container_width=True
)

traffic_data = (
    get_traffic_pattern(df)
)

st.dataframe(
    traffic_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# TRAFFIC FINDING
# ============================================================

st.info(
    """
    **Finding:** Delivery time generally increases as
    road traffic becomes heavier. Jam conditions show
    the highest average delivery time, while Low traffic
    has the lowest.
    """
)


# ============================================================
# WEATHER ANALYSIS
# ============================================================

st.subheader(
    "Weather Conditions vs Delivery Time"
)

st.plotly_chart(
    create_weather_plot(df),
    use_container_width=True
)

weather_data = (
    get_weather_pattern(df)
)

st.dataframe(
    weather_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# WEATHER FINDING
# ============================================================

st.info(
    """
    **Finding:** Sunny conditions have the lowest average
    delivery time, while Cloudy and Fog conditions show
    higher average delivery times.
    """
)


# ============================================================
# TRAFFIC × WEATHER
# ============================================================

st.divider()

st.header(
    "3. Combined Traffic & Weather Analysis"
)

st.markdown(
    """
    This analysis examines delivery-time patterns for
    combinations of road traffic density and weather
    conditions.
    """
)


# ============================================================
# HEATMAP
# ============================================================

st.plotly_chart(
    create_traffic_weather_heatmap(df),
    use_container_width=True
)


# ============================================================
# COMBINED DATA TABLE
# ============================================================

combined_data = (
    get_traffic_weather_pattern(df)
)

st.subheader(
    "Traffic × Weather Data"
)

st.dataframe(
    combined_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# COMBINED FINDINGS
# ============================================================

st.subheader(
    "Combined Pattern Findings"
)

st.info(
    """
    **Key observation:**

    The combination of Jam traffic with Fog or Cloudy
    weather conditions shows some of the highest average
    delivery times in the dataset.

    Jam + Fog has an average delivery time of approximately
    36.81 minutes, while Jam + Cloudy has approximately
    36.69 minutes.

    These are observed associations in the dataset and
    should not be interpreted as causal relationships.
    """
)


# ============================================================
# OVERALL FINDINGS
# ============================================================

st.divider()

st.header(
    "4. Overall Findings"
)

st.markdown(
    """
    ### Temporal Patterns

    - Evening orders around 19:00–21:00 have the highest
      average delivery times.
    - Morning orders around 8:00–10:00 have lower average
      delivery times.

    ### Traffic Patterns

    - Low traffic has the lowest average delivery time.
    - Jam traffic has the highest average delivery time.

    ### Weather Patterns

    - Sunny conditions have lower average delivery times.
    - Cloudy and Fog conditions have higher average
      delivery times.

    ### Combined Pattern

    - High delivery times are particularly visible when
      heavy traffic is combined with adverse weather
      conditions.

    These findings represent associations observed in
    the dataset and should not be interpreted as
    causal relationships.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Food Delivery Pattern Mining • "
    "Member 2: Temporal & Operational Analysis"
)