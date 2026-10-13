import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load cleaned data
df = pd.read_csv("100_back_cleaned.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Extract the course suffix
df["Course"] = (
    df["Race_Time"].astype(str)
    .str.extract(r"([A-Za-z]+)$")[0]
    .str.upper()
)

# Use confirmed long-course meter results only
df = df[df["Course"] == "L"].copy()
df = df.sort_values("Date")

# Choose prediction date
prediction_date = pd.Timestamp("2025-07-24")

# Use previous 12 months for training
start_date = prediction_date - pd.DateOffset(months=3)

train = df[
    (df["Date"] >= start_date) &
    (df["Date"] < prediction_date)
].copy()

test = df[df["Date"] == prediction_date].copy()

# Show which races are used for training
print("\nTraining data:")
print(train[["Date", "Race_Time", "Time_seconds"]])
print("\nNumber of training races:", len(train))

# Convert dates to numerical values
train["Days"] = (
    train["Date"] - start_date
).dt.days

test["Days"] = (
    test["Date"] - start_date
).dt.days

# Train linear regression
model = LinearRegression()
model.fit(train[["Days"]], train["Time_seconds"])

# Predict April time
prediction = model.predict(test[["Days"]])[0]
actual = test["Time_seconds"].iloc[0]

print("Predicted time:", round(prediction, 2))
print("Actual time:", round(actual, 2))
print("Absolute error:", round(abs(prediction - actual), 2))

# Plot results
plt.scatter(
    train["Date"],
    train["Time_seconds"],
    label="Training results"
)

plt.scatter(
    test["Date"],
    test["Time_seconds"],
    label="Actual result"
)

plt.scatter(
    test["Date"],
    [prediction],
    label="Predicted result"
)

# Plot regression line
plot_dates = pd.date_range(
    train["Date"].min(),
    prediction_date,
    periods=100
)

plot_days = (
    plot_dates - start_date
).days.to_numpy().reshape(-1, 1)

plt.plot(
    plot_dates,
    model.predict(plot_days),
    linestyle="--",
    label="Regression trend"
)

plt.xlabel("Date")
plt.ylabel("100 Back time (seconds)")
plt.title("100 Back LCM Race Time Prediction")
plt.legend()
plt.grid(True)

# Slant the dates 45 degrees
plt.xticks(rotation=45, ha="right")

# Automatically adjust spacing
plt.tight_layout()

plt.show()

print("\nComparing training windows:")

results = []

for months in [3, 6, 12]:
    window_start = prediction_date - pd.DateOffset(months=months)

    window_train = df[
        (df["Date"] >= window_start) &
        (df["Date"] < prediction_date)
    ].copy()

    if len(window_train) < 2:
        print(f"{months} months: Not enough data")
        continue

    window_train["Days"] = (
        window_train["Date"] - window_start
    ).dt.days

    target_days = (
        prediction_date - window_start
    ).days

    window_model = LinearRegression()
    window_model.fit(
        window_train[["Days"]],
        window_train["Time_seconds"]
    )

    predicted = window_model.predict(
        pd.DataFrame({"Days": [target_days]})
    )[0]

    error = abs(predicted - actual)

    results.append({
        "Window": f"{months} months",
        "Training races": len(window_train),
        "Predicted time": round(predicted, 2),
        "Actual time": round(actual, 2),
        "Error": round(error, 2)
    })

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

