import pandas as pd

# Read Jessie's swimming results CSV
df = pd.read_csv(
    "SwimingResults_2018-2026_LiJessieYueming (1).csv"
)

# Show the first 5 rows
print(df.head())

# Show all column names
print("\nColumn names:")
print(df.columns.tolist())

columns = [
    "Date",
    "Event",
    "Event Age",
    "Prelim",
    "Finals"
]

filtered_df = df[columns].copy()

# Display the first 10 rows
print(filtered_df.head(10))

# Save the filtered data
filtered_df.to_csv(
    "filtered_swimming_data.csv",
    index=False
)

print("Filtering completed!")

# Filter for 100 Back
backstroke_df = filtered_df[
    filtered_df["Event"] == "100 Back"
].copy()

# Convert dates into proper date format
backstroke_df["Date"] = pd.to_datetime(
    backstroke_df["Date"]
)

# Sort chronologically
backstroke_df = backstroke_df.sort_values("Date")

# Save 100 Back results
backstroke_df.to_csv(
    "100_back_results.csv",
    index=False
)

print(backstroke_df.head(10))
print("Number of results:", len(backstroke_df))

import re

def convert_time(value):
    if pd.isna(value):
        return None

    # Remove letter suffixes
    clean = re.sub(
        r"[A-Za-z]+$", "", str(value).strip()
    )

    parts = clean.split(":")

    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return None

# Use Finals time when available,
# otherwise use Prelim time
backstroke_df["Race_Time"] = (
    backstroke_df["Finals"].fillna(
        backstroke_df["Prelim"]
    )
)

# Convert to seconds
backstroke_df["Time_seconds"] = (
    backstroke_df["Race_Time"].apply(convert_time)
)

print(backstroke_df[
    ["Date", "Event", "Race_Time", "Time_seconds"]
].head(57))

backstroke_df.to_csv(
    "100_back_cleaned.csv",
    index=False
)

print("Cleaned swimming data saved!")