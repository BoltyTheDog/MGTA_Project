import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from Classes.Flight import Flight
import numpy as np
from collections import Counter

def compute_slots(hstart: int, hend: int, hnoreg: float, paar: int, aar: int) -> np.ndarray:
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
    while t < hend:
        slots.append([int(round(t)), 0, 0])
        t += slot_interval_reg

    # Post-regulation period: use AAR
    slot_interval_nom = 60 / aar
    while t < hnoreg:
        slots.append([int(round(t)), 0, 0])
        t += slot_interval_nom

    return np.array(slots, dtype=int)


def initialise_flights(filename: str) -> list['Flight'] | None:
    flights = []
    with open(filename, 'r') as r:
        next(r)  # skip header
        for line in r:
            line_array = line.strip().split(";")
            if line_array[3] == "LEBL":
                dep_time = datetime.strptime(line_array[6], "%H:%M:%S")
                arr_time = datetime.strptime(line_array[8], "%H:%M:%S")
                taxi_time = timedelta(minutes=int(line_array[7]))
                flight_time = datetime.strptime(line_array[10], "%H:%M:%S")
                flight_duration = timedelta(
                    hours=flight_time.hour, minutes=flight_time.minute, seconds=flight_time.second
                )

                # Determine ECAC status (column 15 in your example)
                is_ecac = line_array[14].strip().upper() == "ECAC" if len(line_array) > 14 else True

                flights.append(Flight(
                    line_array[0], line_array[1], line_array[2], line_array[3],
                    int(line_array[5]), float(line_array[16]), dep_time, taxi_time,
                    arr_time, flight_duration, float(line_array[17]), line_array[11], int(line_array[12]),
                    is_ecac
                ))

    return flights if flights else None

#def exempt_flights(flights: list[Flight]) -> list[Flight]:


def amount_flights_by_hour(flights: list[Flight], airline: str, hour1: int, hour2: int) -> int:
    counter = 0
    for f in flights:
        if f.callsign.startswith(airline) and hour1 <= f.arr_time.hour <= hour2:
            counter += 1
    return counter

def flight_by_callsign(flights: list[Flight], callsign: str) -> Flight | None:
    return next((f for f in flights if f.callsign == callsign), None)
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


def plot_aggregated_demand(flights: list[Flight], reghstart: int, reghend: int, max_capacity: int,
                           min_capacity: int) -> float:
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
        hours_range[reg_start_min:reg_end_min + 1],
        reduced_line[reg_start_min:reg_end_min + 1],
        color="red", linestyle="--", linewidth=2, label=f"{min_capacity}/hour capacity reduced"
    )
    y_reduced_end = reduced_line[reg_end_min]

    # Nominal capacity line (garnet dotted) and find intercept
    nominal_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    intercepts = []
    for m in range(reg_end_min + 1, 1440):
        nominal_line[m] = y_reduced_end + max_capacity * ((m - reg_end_min) / 60)
        if nominal_line[m] >= cumulative_arrivals[m]:
            intercepts.append(m)
            if len(intercepts) == 2:
                break

    # Determine intercept hour
    if len(intercepts) >= 2:
        intercept_minute = intercepts[1]
        intercept_hour = intercept_minute / 60
        nominal_line[intercept_minute:] = cumulative_arrivals[intercept_minute:]
        end_minute = intercept_minute
    else:
        intercept_hour = 24.0  # No intercept before end of day
        end_minute = 1439

    plt.plot(
        hours_range[reg_end_min + 1:end_minute + 1],
        nominal_line[reg_end_min + 1:end_minute + 1],
        color="#800000", linestyle=":", linewidth=2, label=f"{max_capacity}/hour capacity nominal"
    )

    # Combine both regulation lines for area calculation
    regulation_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    regulation_line[reg_start_min:reg_end_min + 1] = reduced_line[reg_start_min:reg_end_min + 1]
    regulation_line[reg_end_min + 1:end_minute + 1] = nominal_line[reg_end_min + 1:end_minute + 1]

    # Shade and compute only where regulation line is below the aggregated demand
    mask = (regulation_line[reg_start_min:end_minute + 1] < cumulative_arrivals[reg_start_min:end_minute + 1])
    area = np.trapz(
        cumulative_arrivals[reg_start_min:end_minute + 1][mask] - regulation_line[reg_start_min:end_minute + 1][mask],
        dx=1
    )
    plt.fill_between(
        hours_range[reg_start_min:end_minute + 1],
        regulation_line[reg_start_min:end_minute + 1],
        cumulative_arrivals[reg_start_min:end_minute + 1],
        where=mask,
        color='orange', alpha=0.3, label="Regulation area"
    )

    print(f"Total area where regulation line is below aggregated demand: {area / 60:.2f} plane-hour")

    num_flights = len(flights)
    average_delay_minutes = (area / num_flights) if num_flights > 0 else 0
    print(f"Average delay per flight: {average_delay_minutes:.2f} minutes")
    print(f"Nominal line intercepts with aggregated demand at hour: {intercept_hour:.2f}")

    plt.figtext(0.7, 0.04, f"Average delay per flight: {average_delay_minutes:.2f} minutes",
                ha="center", fontsize=8, color="black")
    plt.figtext(0.7, 0.01, f"Total area where regulation line is below aggregated demand: {area / 60:.2f} plane-hour",
                ha="center", fontsize=8, color="black")
    plt.figtext(0.7, 0.07, f"Nominal line intercept at hour: {intercept_hour:.2f}",
                ha="center", fontsize=8, color="black")

    plt.xlabel("Hour")
    plt.ylabel("Cumulative arrivals")
    plt.title("Aggregated demand - Cumulative arrivals per minute")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.xticks(range(0, 25, 2))
    plt.legend()
    plt.show()

    return intercept_hour


def filter_arrival_flights(arrival_flights: list[Flight],
                           distance_threshold: float,
                           h_start: int,
                           h_no_reg: float,
                           h_file: int) -> list[Flight]:
    """
    Filters arrival flights and updates exemption and delay type attributes:
    - is_exempt = True for non-ECAC flights OR flights meeting exemption criteria
    - delay_type = "None" for exempt flights outside regulation window
    - delay_type = "Air" for exempt flights inside regulation window
    - delay_type = "Ground" for non-exempt flights inside regulation window

    Parameters:
    arrival_flights: list[Flight] - List of Flight objects to filter
    distance_threshold: float - Distance threshold in KM for exemption
    h_start: int - Regulation start hour
    h_no_reg: float - End of regulation hour (from HNoReg)
    h_file: int - Filing hour (publishing time)

    Returns:
    list[Flight] - Same vector with updated is_exempt and delay_type attributes
    """

    def parse_time_to_hours(time_obj: datetime) -> float:
        """Convert datetime object to hours since midnight"""
        return time_obj.hour + time_obj.minute / 60.0 + time_obj.second / 3600.0

    def should_be_exempt(flight: Flight, dist_threshold: float, publishing_time: int) -> bool:
        """
        Determine if flight should be exempt based on:
        1. NON-ECAC flight
        2. Distance > threshold (for ECAC flights)
        3. ETD before publishing time + 30 minutes
        """
        # Condition 1: Check if flight is from NON-ECAC
        if not flight.is_ecac:
            return True

        # Condition 2: Check if distance is larger than threshold (for ECAC flights)
        if flight.flight_distance > dist_threshold:
            return True

        # Condition 3: Check ETD against publishing time
        etd_hours = parse_time_to_hours(flight.dep_time)
        if etd_hours < publishing_time + 0.5:  # 30 minutes buffer
            return True

        return False

    def determine_delay_type(flight: Flight, is_exempt_status: bool, reg_start: int, reg_end: float) -> str:
        """
        Determine delay type based on exemption status and arrival time:
        - Exempt flights: "None" if outside regulation, "Air" if inside regulation
        - Non-exempt flights: "Ground" if inside regulation, "None" if outside regulation
        """
        eta_hours = parse_time_to_hours(flight.arr_time)

        # Check if arrival is within regulation window
        is_inside_regulation = reg_start <= eta_hours < reg_end

        if is_exempt_status:
            # Exempt flights: "Air" delay if inside regulation, "None" if outside
            return "Air" if is_inside_regulation else "None"
        else:
            # Non-exempt flights: "Ground" delay if inside regulation, "None" if outside
            return "Ground" if is_inside_regulation else "None"

    # Process each flight
    for flight in arrival_flights:
        # Determine exemption status (excluding arrival time check)
        is_exempt_status = should_be_exempt(flight, distance_threshold, h_file)

        # Determine delay type based on exemption status and arrival time
        delay_type = determine_delay_type(flight, is_exempt_status, h_start, h_no_reg)

        # Update flight attributes
        flight.is_exempt = is_exempt_status
        flight.delay_type = delay_type

    # Print summary statistics
    ecac_count = sum(1 for f in arrival_flights if f.is_ecac)
    non_ecac_count = len(arrival_flights) - ecac_count
    exempt_count = sum(1 for f in arrival_flights if f.is_exempt)
    non_exempt_count = len(arrival_flights) - exempt_count

    # Count delay types
    delay_types = {
        "None": 0,
        "Ground": 0,
        "Air": 0
    }
    for flight in arrival_flights:
        delay_types[flight.delay_type] = delay_types.get(flight.delay_type, 0) + 1

    print(f"Processed {len(arrival_flights)} flights")
    print(f"ECAC flights: {ecac_count}")
    print(f"Non-ECAC flights: {non_ecac_count}")
    print(f"Exempt flights: {exempt_count}")
    print(f"Non-exempt flights: {non_exempt_count}")
    print(f"Delay types - None: {delay_types['None']}, Ground: {delay_types['Ground']}, Air: {delay_types['Air']}")

    return arrival_flights  # Return the same updated vector

if __name__ == "__main__":
    print("You're not executing the main program")