import pandas as pd
from pathlib import Path


BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")

INPUT_FILE = BASE_DIR / "data" / "processed" / "queue_wait_analysis_jan2024.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "driver_repositioning_recommendations_jan2024.csv"


def calculate_required_extra_drivers(passenger_rate, driver_rate, safety_buffer=1):
    """
    Minimum extra drivers needed so driver_rate > passenger_rate.
    safety_buffer adds extra cushion so the system is not barely stable.
    """

    if driver_rate > passenger_rate:
        return 0

    required_driver_rate = passenger_rate + safety_buffer
    extra_drivers_needed = required_driver_rate - driver_rate

    return max(0, extra_drivers_needed)


def optimize_driver_repositioning():
    df = pd.read_csv(INPUT_FILE)

    df["extra_drivers_needed"] = df.apply(
        lambda row: calculate_required_extra_drivers(
            row["passenger_arrivals"],
            row["driver_activity"],
            safety_buffer=1
        ),
        axis=1
    )

    df["recommended_action"] = df["extra_drivers_needed"].apply(
        lambda x: "Reposition additional drivers" if x > 0 else "No action needed"
    )

    shortage_df = df[df["extra_drivers_needed"] > 0].copy()

    shortage_df = shortage_df.sort_values(
        by="extra_drivers_needed",
        ascending=False
    )

    shortage_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved optimization recommendations to: {OUTPUT_FILE}")

    print(shortage_df[[
        "pickup_hour",
        "Borough",
        "Zone",
        "passenger_arrivals",
        "driver_activity",
        "extra_drivers_needed",
        "recommended_action"
    ]].head(20))


if __name__ == "__main__":
    optimize_driver_repositioning()