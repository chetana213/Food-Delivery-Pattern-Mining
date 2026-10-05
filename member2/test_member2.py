import pandas as pd

from temporal_module import (
    prepare_temporal_features,
    get_hourly_delivery_pattern,
    get_day_of_week_pattern,
    get_weekend_pattern
)

from operational_module import (
    get_traffic_pattern,
    get_weather_pattern,
    get_traffic_weather_pattern
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "Zomato Dataset.csv"
)

print("\nDATASET")
print("----------------")

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# PREPARE TEMPORAL FEATURES
# ============================================================

df = prepare_temporal_features(df)


print("\nTEMPORAL FEATURES")
print("------------------")

print(
    df[
        [
            "Order_Hour",
            "Day_of_Week",
            "Weekend",
            "Month"
        ]
    ].head()
)


# ============================================================
# HOURLY PATTERN
# ============================================================

print("\nHOURLY DELIVERY PATTERN")
print("------------------------")

hourly_data = get_hourly_delivery_pattern(df)

print(hourly_data)


# ============================================================
# DAY OF WEEK PATTERN
# ============================================================

print("\nDAY OF WEEK PATTERN")
print("-------------------")

day_data = get_day_of_week_pattern(df)

print(day_data)


# ============================================================
# WEEKEND PATTERN
# ============================================================

print("\nWEEKEND VS WEEKDAY")
print("------------------")

weekend_data = get_weekend_pattern(df)

print(weekend_data)


# ============================================================
# TRAFFIC PATTERN
# ============================================================

print("\nTRAFFIC PATTERN")
print("----------------")

traffic_data = get_traffic_pattern(df)

print(traffic_data)


# ============================================================
# WEATHER PATTERN
# ============================================================

print("\nWEATHER PATTERN")
print("----------------")

weather_data = get_weather_pattern(df)

print(weather_data)


# ============================================================
# TRAFFIC × WEATHER
# ============================================================

print("\nTRAFFIC × WEATHER PATTERN")
print("--------------------------")

traffic_weather_data = (
    get_traffic_weather_pattern(df)
)

print(traffic_weather_data)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print(
    "\nMEMBER 2 TEST COMPLETED SUCCESSFULLY"
)