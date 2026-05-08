import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\SaiKrishna\OneDrive\Desktop\queuing project")

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YELLOW_FILE = RAW_DIR / "yellow_tripdata_2024-01.parquet"
FHV_FILE = RAW_DIR / "fhv_tripdata_2024-01.parquet"
ZONE_FILE = RAW_DIR / "taxi_zone_lookup.csv"


def build_passenger_arrivals():
    df = pd.read_parquet(YELLOW_FILE)

    print("Yellow columns:", df.columns.tolist())

    df["pickup_hour"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.floor("h")

    passenger_arrivals = (
        df.groupby(["pickup_hour", "PULocationID"])
        .size()
        .reset_index(name="passenger_arrivals")
    )

    passenger_arrivals = passenger_arrivals.rename(
        columns={"PULocationID": "location_id"}
    )

    print("Passenger columns:", passenger_arrivals.columns.tolist())

    return passenger_arrivals


def build_driver_activity():
    df = pd.read_parquet(FHV_FILE)

    print("FHV columns:", df.columns.tolist())

    df["pickup_hour"] = pd.to_datetime(df["pickup_datetime"]).dt.floor("h")

    # FHV sometimes uses PUlocationID, sometimes PULocationID
    if "PUlocationID" in df.columns:
        location_col = "PUlocationID"
    elif "PULocationID" in df.columns:
        location_col = "PULocationID"
    else:
        raise KeyError("Could not find pickup location column in FHV file.")

    driver_activity = (
        df.groupby(["pickup_hour", location_col])
        .size()
        .reset_index(name="driver_activity")
    )

    driver_activity = driver_activity.rename(
        columns={location_col: "location_id"}
    )

    print("Driver columns:", driver_activity.columns.tolist())

    return driver_activity


def build_queue_features():
    passengers = build_passenger_arrivals()
    drivers = build_driver_activity()

    print("Before merge - passengers:", passengers.columns.tolist())
    print("Before merge - drivers:", drivers.columns.tolist())

    features = passengers.merge(
        drivers,
        on=["pickup_hour", "location_id"],
        how="outer"
    ).fillna(0)

    features["imbalance_ratio"] = (
        features["passenger_arrivals"] / (features["driver_activity"] + 1)
    )

    features["net_demand"] = (
        features["passenger_arrivals"] - features["driver_activity"]
    )

    zones = pd.read_csv(ZONE_FILE)
    zones = zones.rename(columns={"LocationID": "location_id"})

    features = features.merge(zones, on="location_id", how="left")

    output_path = PROCESSED_DIR / "queue_features_jan2024.csv"
    features.to_csv(output_path, index=False)

    print(f"Saved queue features to: {output_path}")
    print(features.head())


if __name__ == "__main__":
    build_queue_features()