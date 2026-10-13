import pandas as pd

# Load cleaned swimming data
df = pd.read_csv("100_back_cleaned.csv")

# Convert dates
df["Date"] = pd.to_datetime(df["Date"])

# Identify course from race time suffix
df["Course"] = (
    df["Race_Time"]
    .astype(str)
    .str.extract(r"([A-Za-z]+)$")[0]
    .str.upper()
)

# Check course types
print("Course types found:")
print(df["Course"].value_counts(dropna=False))

# Separate yards and long-course meters
yards = df[df["Course"] == "Y"].copy()
meters = df[df["Course"] == "L"].copy()

# Save separate datasets
yards.to_csv("100_back_yards.csv", index=False)
meters.to_csv("100_back_meters.csv", index=False)

print("\nYards races:", len(yards))
print("Meters races:", len(meters))

# Show records with unexpected suffixes
unknown = df[~df["Course"].isin(["Y", "L"])]

print("\nRecords needing review:")
print(unknown[["Date", "Race_Time", "Course"]])

# Count races by year and course
df["Year"] = df["Date"].dt.year

summary = pd.crosstab(
    df["Year"],
    df["Course"]
)

print("\nRaces by year and course:")
print(summary)

# Show all 2025 and 2026 races
recent = df[df["Year"].isin([2025, 2026])]

print("\nRecent races:")
print(
    recent[
        ["Date", "Race_Time", "Time_seconds", "Course"]
    ].to_string(index=False)
)
