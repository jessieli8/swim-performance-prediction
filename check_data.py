import pandas as pd

df = pd.read_csv("100_back_cleaned.csv")

# Check the data structure
print("Total rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())

# Check how many valid race times we have
print("\nValid race times:")
print(df["Time_seconds"].notna().sum())

# Check the date range
df["Date"] = pd.to_datetime(df["Date"])

print("\nEarliest race:", df["Date"].min())
print("Latest race:", df["Date"].max())

# Check the number of races each year
print("\nRaces per year:")
print(df.groupby(df["Date"].dt.year).size())

# Show sample results
print("\nSample data:")
print(df[["Date", "Event", "Race_Time", "Time_seconds"]].head(10))
