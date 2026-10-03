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
# 1. LOAD DATASET
# ============================================================

df = load_dataset("Zomato Dataset.csv")


# ============================================================
# 2. DATASET OVERVIEW
# ============================================================

print("\nDATASET OVERVIEW")
print("----------------")

overview = get_dataset_overview(df)

print("Total records:", overview["rows"])
print("Total attributes:", overview["columns"])
print("Duplicate records:", overview["duplicates"])
print("Start date:", overview["start_date"])
print("End date:", overview["end_date"])


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("----------------")

missing_values = get_missing_values(df)

print(missing_values)


# ============================================================
# 4. DATA TYPES
# ============================================================

print("\nDATA TYPES")
print("----------------")

print(get_data_types(df))


# ============================================================
# 5. DATA PREVIEW
# ============================================================

print("\nDATA PREVIEW")
print("----------------")

print(get_dataset_preview(df))


# ============================================================
# 6. CALCULATE DELIVERY DISTANCE
# ============================================================

df = calculate_delivery_distance(df)


# ============================================================
# 7. CREATE VALID DISTANCE DATASET
# ============================================================

df_distance = get_distance_dataset(df)


print("\nDISTANCE DATASET")
print("----------------")

print("Original records:", len(df))
print("Valid distance records:", len(df_distance))
print("Excluded records:", len(df) - len(df_distance))


# ============================================================
# 8. DISTANCE STATISTICS
# ============================================================

print("\nDISTANCE STATISTICS")
print("-------------------")

distance_stats = get_distance_statistics(df_distance)

for key, value in distance_stats.items():
    print(f"{key}: {value}")


# ============================================================
# 9. DISTANCE-TIME CORRELATION
# ============================================================

print("\nDISTANCE-TIME CORRELATION")
print("-------------------------")

correlation = get_distance_time_correlation(df_distance)

print(f"Pearson correlation: {correlation:.6f}")


# ============================================================
# 10. INTERACTIVE DISTANCE VS TIME PLOT
# ============================================================

fig = create_distance_time_plot(df_distance)

fig.show()