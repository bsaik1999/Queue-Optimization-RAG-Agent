import pandas as pd
from pathlib import Path


BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")
INPUT_FILE = BASE_DIR / "data" / "processed" / "queue_features_jan2024.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "queue_wait_analysis_jan2024.csv"


def estimate_wait_time(passenger_rate, driver_rate):
    if driver_rate <= passenger_rate:
        return float("inf")

    wait_time = passenger_rate / (driver_rate * (driver_rate - passenger_rate))
    return wait_time


def estimate_queue_status(passenger_rate, driver_rate):
    if driver_rate == 0 and passenger_rate > 0:
        return "No driver supply"
    elif driver_rate <= passenger_rate:
        return "Unstable / shortage"
    else:
        return "Stable"


def build_wait_time_analysis():
    df = pd.read_csv(INPUT_FILE)

    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"], errors="coerce")

    df = df[
        (df["pickup_hour"] >= "2024-01-01") &
        (df["pickup_hour"] < "2024-02-01")
    ]

    df["estimated_wait_time"] = df.apply(
        lambda row: estimate_wait_time(
            row["passenger_arrivals"],
            row["driver_activity"]
        ),
        axis=1
    )

    df["queue_status"] = df.apply(
        lambda row: estimate_queue_status(
            row["passenger_arrivals"],
            row["driver_activity"]
        ),
        axis=1
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved wait-time analysis to: {OUTPUT_FILE}")

    print(df[[
        "pickup_hour",
        "location_id",
        "Borough",
        "Zone",
        "passenger_arrivals",
        "driver_activity",
        "imbalance_ratio",
        "net_demand",
        "estimated_wait_time",
        "queue_status"
    ]].head(20))


if __name__ == "__main__":
    build_wait_time_analysis()