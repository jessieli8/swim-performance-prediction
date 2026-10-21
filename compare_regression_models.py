
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load Jessie's cleaned data
df = pd.read_csv("100_back_cleaned.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Keep confirmed long-course meter results
df["Course"] = (
    df["Race_Time"].astype(str)
    .str.extract(r"([A-Za-z]+)$")[0]
    .str.upper()
)

df = df[df["Course"] == "L"].copy()
df = df.sort_values("Date")

# Predict the same race as before
prediction_date = pd.Timestamp("2025-07-24")

# Change this to experiment with different windows
months = 6

start_date = prediction_date - pd.DateOffset(months=months)

train = df[
    (df["Date"] >= start_date) &
    (df["Date"] < prediction_date)
].copy()

test = df[df["Date"] == prediction_date].copy()

if len(train) < 3 or len(test) != 1:
    raise ValueError("Not enough training or test data")

# Convert dates to days since start of window
train["Days"] = (train["Date"] - start_date).dt.days
test["Days"] = (test["Date"] - start_date).dt.days

X_train = train[["Days"]]
y_train = train["Time_seconds"]

X_test = test[["Days"]]
actual = test["Time_seconds"].iloc[0]

# Model 1: Linear regression
linear = LinearRegression()
linear.fit(X_train, y_train)

# Model 2: Polynomial regression
polynomial = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
polynomial.fit(X_train, y_train)

# Model 3: Exponential regression
# Fit log(time) as a linear function of days
exponential = LinearRegression()
exponential.fit(X_train, np.log(y_train))

# Generate predictions
predictions = {
    "Linear": linear.predict(X_test)[0],
    "Polynomial": polynomial.predict(X_test)[0],
    "Exponential": np.exp(
        exponential.predict(X_test)[0]
    )
}

print(f"\nHistorical window: {months} months")
print(f"Training races: {len(train)}")
print(f"Actual race time: {actual:.2f} seconds\n")

for name, predicted in predictions.items():
    error = abs(predicted - actual)
    print(
        f"{name}: {predicted:.2f} sec "
        f"| Error: {error:.2f} sec"
    )

# Plot curves
plot_days = np.linspace(
    0,
    X_test["Days"].iloc[0],
    150
).reshape(-1, 1)

plot_X = pd.DataFrame(plot_days, columns=["Days"])

plot_dates = (
    start_date +
    pd.to_timedelta(plot_days.flatten(), unit="D")
)

plt.scatter(
    train["Date"], y_train,
    label="Training races", color="black"
)

plt.scatter(
    test["Date"], [actual],
    label="Actual result", color="red", s=80
)

plt.plot(
    plot_dates,
    linear.predict(plot_X),
    label="Linear"
)

plt.plot(
    plot_dates,
    polynomial.predict(plot_X),
    label="Polynomial"
)

plt.plot(
    plot_dates,
    np.exp(exponential.predict(plot_X)),
    label="Exponential"
)

plt.xlabel("Date")
plt.ylabel("100 Back LCM time (seconds)")
plt.title(f"Regression Comparison ({months} Months)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(
    start_date - pd.Timedelta(days=10),
    prediction_date + pd.Timedelta(days=10)
)
plt.margins(y=0.1)
plt.show()
