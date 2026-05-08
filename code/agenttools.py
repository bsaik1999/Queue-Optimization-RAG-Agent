import math


def estimate_wait_time(passenger_rate, driver_rate):
    """
    M/M/1-style wait-time approximation.
    passenger_rate = demand arrival rate
    driver_rate = supply/service rate
    """

    if driver_rate <= passenger_rate:
        return float("inf")

    return passenger_rate / (driver_rate * (driver_rate - passenger_rate))


def estimate_queue_status(passenger_rate, driver_rate):
    if driver_rate == 0 and passenger_rate > 0:
        return "No driver supply"
    elif driver_rate <= passenger_rate:
        return "Unstable / shortage"
    else:
        return "Stable"


def calculate_extra_drivers_needed(passenger_rate, driver_rate, safety_buffer=1):
    """
    Minimum extra driver-side capacity needed so driver_rate > passenger_rate.
    """

    if driver_rate > passenger_rate:
        return 0

    required_driver_rate = passenger_rate + safety_buffer
    extra_drivers_needed = required_driver_rate - driver_rate

    return max(0, extra_drivers_needed)


def simulate_driver_increase(passenger_rate, driver_rate, percent_increase):
    """
    Simulates what happens if driver activity increases by a percentage.
    """

    new_driver_rate = driver_rate * (1 + percent_increase / 100)

    old_wait_time = estimate_wait_time(passenger_rate, driver_rate)
    new_wait_time = estimate_wait_time(passenger_rate, new_driver_rate)

    old_status = estimate_queue_status(passenger_rate, driver_rate)
    new_status = estimate_queue_status(passenger_rate, new_driver_rate)

    extra_drivers_needed_after = calculate_extra_drivers_needed(
        passenger_rate,
        new_driver_rate
    )

    return {
        "passenger_rate": passenger_rate,
        "original_driver_rate": driver_rate,
        "percent_driver_increase": percent_increase,
        "new_driver_rate": round(new_driver_rate, 2),
        "old_wait_time": old_wait_time,
        "new_wait_time": new_wait_time,
        "old_queue_status": old_status,
        "new_queue_status": new_status,
        "extra_drivers_needed_after_increase": math.ceil(extra_drivers_needed_after)
    }


def analyze_queue_scenario(passenger_rate, driver_rate):
    """
    Full queue diagnosis for one zone-hour scenario.
    """

    wait_time = estimate_wait_time(passenger_rate, driver_rate)
    status = estimate_queue_status(passenger_rate, driver_rate)
    extra_drivers = calculate_extra_drivers_needed(passenger_rate, driver_rate)

    return {
        "passenger_rate": passenger_rate,
        "driver_rate": driver_rate,
        "estimated_wait_time": wait_time,
        "queue_status": status,
        "extra_drivers_needed": math.ceil(extra_drivers)
    }


if __name__ == "__main__":
    print("Base scenario:")
    print(analyze_queue_scenario(passenger_rate=726, driver_rate=4))

    print("\nWhat-if scenario:")
    print(simulate_driver_increase(
        passenger_rate=726,
        driver_rate=4,
        percent_increase=20
    ))