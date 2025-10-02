import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from Classes.Flight import Flight
import numpy as np
from collections import Counter

def compute_slots(hstart: int, hend: int, hnoreg: int, paar: int, aar: int) -> np.ndarray:
    """
    Computes the slot matrix for airport regulation.
    Each slot: [slot_time, flight_id, airline_id]
    - slot_time: initial time of the slot (in minutes)
    - flight_id: initially 0
    - airline_id: initially 0

    Parameters:
    Hstart: int, start time in minutes
    Hend: int, end time in minutes
    HNoReg: int, end of regulation in minutes
    PAAR: int, reduced capacity (slots per hour) during regulation
    AAR: int, nominal capacity (slots per hour) after regulation

    Returns:
    slots: np.ndarray, shape (n_slots, 3)
    """
    slots = []
    t = hstart

    # Regulation period: use PAAR
    slot_interval_reg = 60 / paar
    while t < hnoreg:
        slots.append([int(round(t)), 0, 0])
        t += slot_interval_reg

    # Post-regulation period: use AAR
    slot_interval_nom = 60 / aar
    while t < hend:
        slots.append([int(round(t)), 0, 0])
        t += slot_interval_nom

    return np.array(slots, dtype=int)

def initialise_flights(filename: str) -> list['Flight'] | None:
    flights = []
    with open(filename, 'r') as r:
        next(r)  # skip header
        for line in r:
            line_array = line.strip().split(",")  # remove newlines
            if line_array[3] == "LEBL":
                dep_time = datetime.strptime(line_array[6], "%H:%M:%S")
                arr_time = datetime.strptime(line_array[8], "%H:%M:%S")
                taxi_time = timedelta(minutes=int(line_array[7]))
                flight_time = datetime.strptime(line_array[10], "%H:%M:%S") 
                flight_duration = timedelta(
                    hours=flight_time.hour, minutes=flight_time.minute, seconds=flight_time.second
                )

                flights.append(Flight(
                    line_array[0], line_array[1], line_array[2], line_array[3],
                    line_array[4], int(line_array[5]), dep_time, taxi_time,
                    arr_time, flight_duration, line_array[11], int(line_array[12])
                ))

    return flights if flights else None


def amount_flights_by_hour(flights: list[Flight], airline: str, hour1: int, hour2: int) -> int:
    counter = 0
    for f in flights:
        if f.callsign.startswith(airline) and hour1 <= f.arr_time.hour <= hour2:
            counter = counter + 1
    return counter

def flight_by_callsign(flights: list[Flight], callsign: str) -> Flight | None:
    for f in flights:
        if f.callsign == callsign:
            return f
    else:
        return None
"""
def flight_distance(f: Flight) -> float:
    velocity = 500   #f.speed
    time = f.flight_time
    distance = velocity * time.total_seconds()
    return distance
"""
""" def flight_velocity_kt(f: Flight) -> float:
    fl = f.cruise
    if fl == 80:
        return 280
    elif fl = 90:
        return 
"""


def excempt_flights(flights: list[Flight]) -> list[Flight]:
    excemptflights = []
    for flight in flights:
        if flight.exempt:
            excemptflights.append(flight)
    return excemptflights

def plot_flight_count(flights: list[Flight], max_capacity: int, reghstart: int, reghend: int) -> None:

    if reghstart < 0:
        reghstart = 0
    if reghend > 24:
        reghend = 24
    hours = [f.arr_time.hour for f in flights]

    counter = Counter(hours)
    print("Contador de vuelos por hora:")
    for h, c in sorted(counter.items()):
        print(f"{h}:00 -> {c} vuelos")

    plt.hist(hours, bins=24, range=(0,24), color="navy", edgecolor="black")

    plt.xlabel("Hour")
    plt.ylabel("Number of arrivals")
    plt.title("Arrivals per hour")

    plt.hlines(y=max_capacity, xmin=0, xmax=reghstart, colors="green", linewidth=2)
    plt.hlines(y=(max_capacity / 2), xmin=reghstart, xmax=reghend, colors="red", linewidth=2)
    plt.hlines(y=max_capacity, xmin=reghend, xmax=24, colors="green", linewidth=2)

    plt.show()
    return None

def plot_aggregated_demand(flights: list[Flight], reghstart: int, reghend: int, max_capacity: int, min_capacity: int) -> None:
    # Convert arrival times to minutes since midnight
    minutes = [f.arr_time.hour * 60 + f.arr_time.minute for f in flights]
    counter = Counter(minutes)

    # Create arrays for all 1440 minutes in a day (24 * 60)
    minutes_range = np.arange(1440, dtype=int)  # 0 to 1439 minutes
    arrivals_per_minute = np.array([counter.get(m, 0) for m in minutes_range])

    # Calculate cumulative arrivals
    cumulative_arrivals = np.cumsum(arrivals_per_minute)
    hours_range = minutes_range / 60

    plt.figure(figsize=(10, 6))
    plt.plot(hours_range, cumulative_arrivals, linewidth=2, color="#0A7700", label="Aggregated demand")

    # Regulation lines
    reg_start_min = int(reghstart * 60)
    reg_end_min = int(reghend * 60)
    y_start = cumulative_arrivals[reg_start_min]

    # Reduced capacity line (red dashed)
    reduced_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    for m in range(reg_start_min, reg_end_min + 1):
        reduced_line[m] = y_start + min_capacity * ((m - reg_start_min) / 60)
    plt.plot(
        hours_range[reg_start_min:reg_end_min+1],
        reduced_line[reg_start_min:reg_end_min+1],
        color="red", linestyle="--", linewidth=2, label=f"{min_capacity}/hour capacity reduced"
    )
    y_reduced_end = reduced_line[reg_end_min]

    # Nominal capacity line (garnet dotted)
    nominal_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    intercepts = []
    for m in range(reg_end_min + 1, 1440):
        nominal_line[m] = y_reduced_end + max_capacity * ((m - reg_end_min) / 60)
        if nominal_line[m] >= cumulative_arrivals[m]:
            intercepts.append(m)
            if len(intercepts) == 2:
                break
    if len(intercepts) == 2:
        intercept_minute = intercepts[1]
        nominal_line[intercept_minute:] = cumulative_arrivals[intercept_minute:]
        end_minute = intercept_minute
    else:
        end_minute = 1439

    plt.plot(
        hours_range[reg_end_min+1:end_minute+1],
        nominal_line[reg_end_min+1:end_minute+1],
        color="#800000", linestyle=":", linewidth=2, label=f"{max_capacity}/hour capacity nominal"
    )

    # Combine both regulation lines for area calculation
    regulation_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    regulation_line[reg_start_min:reg_end_min+1] = reduced_line[reg_start_min:reg_end_min+1]
    regulation_line[reg_end_min+1:end_minute+1] = nominal_line[reg_end_min+1:end_minute+1]

    # Shade and compute only where regulation line is below the aggregated demand
    mask = (regulation_line[reg_start_min:end_minute+1] < cumulative_arrivals[reg_start_min:end_minute+1])
    area = np.trapz(
        cumulative_arrivals[reg_start_min:end_minute+1][mask] - regulation_line[reg_start_min:end_minute+1][mask],
        dx=1
    )
    plt.fill_between(
        hours_range[reg_start_min:end_minute+1],
        regulation_line[reg_start_min:end_minute+1],
        cumulative_arrivals[reg_start_min:end_minute+1],
        where=mask,
        color='orange', alpha=0.3, label="Regulation area"
    )

    print(f"Total area where regulation line is below aggregated demand: {area/60:.2f} plane-hour")

    num_flights = len(flights)
    average_delay_minutes = (area / num_flights) if num_flights > 0 else 0
    print(f"Average delay per flight: {average_delay_minutes:.2f} minutes")
    plt.figtext(0.7, 0.04, f"Average delay per flight: {average_delay_minutes:.2f} minutes",
                ha="center", fontsize=8, color="black")

    plt.xlabel("Hour")
    plt.ylabel("Cumulative arrivals")
    plt.title("Aggregated demand - Cumulative arrivals per minute")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 2))
    plt.legend()
    # Add the comment below the plot
    plt.figtext(0.7, 0.01, f"Total area where regulation line is below aggregated demand: {area/60:.2f} plane-hour",
                ha="center", fontsize=8, color="black")
    plt.show()
    return None

if __name__ == "__main__":
    print("You're not executing the main program")