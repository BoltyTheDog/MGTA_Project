from typing import Literal

import matplotlib.pyplot as plt  # import for plots
from datetime import datetime,timedelta  # import for 'timedeltas' (OPERATIONS WITH DATETIME OBJECTS)

import pandas as pd  # import of the panda dictionary GHP FUNCTION

from Classes.Flight import Flight  # import of the CLASS FLIGHT and its attributes

from collections import Counter

# imports for linear programming GHP algorithm WP3 (and numpy also for other matters as plots...)
import numpy as np
from scipy.optimize import linprog  # LINEAR PROGRAMMING LIBRARY


def compute_slots(hstart: int, hend: int, hnoreg: float, paar: int, aar: int) -> np.ndarray:
    """
    Computes the slot matrix for airport regulation.
    Each slot: [slot_time, flight_id, airline_id, air_delay, ground_delay]
    - slot_time: initial time of the slot (in minutes)
    - flight_id: initially 0
    - airline_id: initially 0
    - air_delay: initially 0 (in minutes)
    - ground_delay: initially 0 (in minutes)

    Parameters:
    Hstart: int, start time in minutes (regulation start)
    Hend: int, end time in minutes (regulation end, e.g., 18:00 = 1080)
    HNoReg: float, end of all regulation in minutes (when capacity returns to normal completely)
    PAAR: int, reduced capacity (slots per hour) during regulation
    AAR: int, nominal capacity (slots per hour) after regulation

    Returns:
    slots: np.ndarray, shape (n_slots, 5)
    """
    slots = []
    t = hstart

    # Regulation period: use PAAR (reduced capacity)
    slot_interval_reg = 60 / paar
    while t < hend:
        slots.append([int(round(t)), 0, 0, 0, 0])
        t += slot_interval_reg

    # Post-regulation period: use AAR (nominal capacity) - this is key for absorbing delayed flights
    slot_interval_nom = 60 / aar
    while t < hnoreg:
        slots.append([int(round(t)), 0, 0, 0, 0])
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

                # Determine ECAC status

                is_ecac = line_array[14].strip().upper() == "ECAC" if len(line_array) > 14 else True

                flights.append(Flight(
                    line_array[0], line_array[1], line_array[2], line_array[3],
                    int(line_array[5]), float(line_array[16]), dep_time, taxi_time,
                    arr_time, flight_duration, float(line_array[17]), line_array[11], int(line_array[12]),
                    line_array[4], is_ecac
                ))

    return flights if flights else None


def amount_flights_by_hour(flights: list[Flight], airline: str, hour1: int, hour2: int) -> int:
    counter = 0
    for f in flights:
        if f.callsign.startswith(airline) and hour1 <= f.arr_time.hour < hour2:
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


def plot_slotted_arrivals(slotted_flights: list[Flight], max_capacity: int, reghstart: int, reghend: int) -> None:
    """
    Plot histogram of arrivals per hour after slot assignment (GDP implementation).
    Shows the redistributed arrival times compared to capacity constraints.
    
    Parameters:
    slotted_flights: list[Flight] - Flights with assigned slots
    max_capacity: int - Maximum airport capacity per hour
    reghstart: int - Regulation start hour
    reghend: int - Regulation end hour
    """
    
    if reghstart < 0:
        reghstart = 0
    if reghend > 24:
        reghend = 24
    
    # Extract arrival hours from assigned slot times
    slotted_hours = []
    for flight in slotted_flights:
        if hasattr(flight, 'assigned_slot_time') and flight.assigned_slot_time is not None:
            # Convert minutes to hours
            slot_hour = flight.assigned_slot_time // 60
            slotted_hours.append(slot_hour)
        else:
            # Use original arrival time if no slot assigned
            slotted_hours.append(flight.arr_time.hour)
    
    counter = Counter(slotted_hours)
    print("\nContador de vuelos por hora (DESPUÉS del slotting):")
    for h, c in sorted(counter.items()):
        print(f"{h}:00 -> {c} vuelos")
    
    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(slotted_hours, bins=24, range=(0,24), color="darkblue", edgecolor="black", alpha=0.7)
    
    plt.xlabel("Hour")
    plt.ylabel("Number of arrivals")
    plt.title("Arrivals per hour AFTER slot assignment (GDP)")
    
    # Add capacity lines
    plt.hlines(y=max_capacity, xmin=0, xmax=reghstart, colors="green", linewidth=2, label=f"Normal capacity ({max_capacity})")
    plt.hlines(y=(max_capacity / 2), xmin=reghstart, xmax=reghend, colors="red", linewidth=2, label=f"Reduced capacity ({max_capacity // 2})")
    plt.hlines(y=max_capacity, xmin=reghend, xmax=24, colors="green", linewidth=2)
    
    # Place legend at 60% horizontal and 30% vertical of the chart
    plt.legend(loc='upper left', bbox_to_anchor=(0.45, 0.3))
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate and print statistics
    total_flights = len(slotted_flights)
    flights_in_regulation = sum(1 for h in slotted_hours if reghstart <= h < reghend)
    flights_before_regulation = sum(1 for h in slotted_hours if h < reghstart)
    flights_after_regulation = sum(1 for h in slotted_hours if h >= reghend)
    
    print(f"\nStatistics after slot assignment:")
    print(f"  Total flights: {total_flights}")
    print(f"  Flights before regulation ({reghstart}:00): {flights_before_regulation}")
    print(f"  Flights during regulation ({reghstart}:00-{reghend}:00): {flights_in_regulation}")
    print(f"  Flights after regulation (≥{reghend}:00): {flights_after_regulation}")
    
    return None


def compute_hnoreg(flights: list[Flight], reghstart: int, reghend: int, max_capacity: int,
                   min_capacity: int) -> float:
    """
    Compute HNoReg (intercept hour) without plotting.
    
    Parameters:
    flights: list[Flight] - List of flights to analyze
    reghstart: int - Regulation start hour
    reghend: int - Regulation end hour
    max_capacity: int - Maximum airport capacity per hour
    min_capacity: int - Reduced capacity during regulation
    
    Returns:
    float: HNoReg (intercept hour when nominal capacity line meets aggregated demand)
    """
    # Convert arrival times to minutes since midnight
    minutes = [f.arr_time.hour * 60 + f.arr_time.minute for f in flights]
    counter = Counter(minutes)

    # Create arrays for all 1440 minutes in a day (24 * 60)
    minutes_range = np.arange(1440, dtype=int)  # 0 to 1439 minutes
    arrivals_per_minute = np.array([counter.get(m, 0) for m in minutes_range])

    # Calculate cumulative arrivals
    cumulative_arrivals = np.cumsum(arrivals_per_minute)

    # Regulation lines
    reg_start_min = int(reghstart * 60)
    reg_end_min = int(reghend * 60)
    y_start = cumulative_arrivals[reg_start_min]

    # Reduced capacity line
    reduced_line = np.full_like(cumulative_arrivals, np.nan, dtype=float)
    for m in range(reg_start_min, reg_end_min + 1):
        reduced_line[m] = y_start + min_capacity * ((m - reg_start_min) / 60)
    y_reduced_end = reduced_line[reg_end_min]

    # Nominal capacity line and find intercept
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
    else:
        intercept_hour = 24.0  # No intercept before end of day

    return intercept_hour


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
        if etd_hours < publishing_time + (1/3):  # 30 minutes buffer
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


def assignSlotsGDP(filtered_arrivals: list[Flight], slots: np.ndarray) -> list[Flight]:
    """
    Assigns slots to flights according to Ground Delay Program (GDP) rules:
    1. Only assigns slots to flights that need regulation (delay_type = "Air" or "Ground")
    2. Exempt flights with "Air" delay are assigned first (with air delay)
    3. Controlled flights with "Ground" delay are assigned second (with ground delay)
    4. No flight can be assigned a slot before its original ETA
    5. Flights with delay_type = "None" are not assigned slots (they arrive as scheduled)
    
    Parameters:
    filtered_arrivals: list[Flight] - List of flights to assign slots to
    slots: np.ndarray - Slots matrix [slot_time, flight_id, airline_id, air_delay, ground_delay]
    
    Returns:
    list[Flight] - Updated list of flights with assigned slots and delays
    """
    
    def time_to_minutes(time_obj: datetime) -> int:
        """Convert datetime object to minutes since midnight"""
        return time_obj.hour * 60 + time_obj.minute

    
    # Create a copy of the slots matrix to work with
    available_slots = slots.copy()
    
    # Create copies of flights to avoid modifying the original list
    slotted_arrivals = list(filtered_arrivals)
    
    # Filter flights that actually need slot assignment (only those with Air or Ground delay types)
    flights_needing_slots = [f for f in slotted_arrivals if f.delay_type in ["Air", "Ground"]]
    flights_not_needing_slots = [f for f in slotted_arrivals if f.delay_type == "None"]
    
    print(f"Total flights: {len(slotted_arrivals)}")
    print(f"Flights needing slot assignment: {len(flights_needing_slots)} (Air: {sum(1 for f in flights_needing_slots if f.delay_type == 'Air')}, Ground: {sum(1 for f in flights_needing_slots if f.delay_type == 'Ground')})")
    print(f"Flights not needing slots (arriving as scheduled): {len(flights_not_needing_slots)}")
    
    # Add new attributes to track slot assignment for all flights
    for flight in slotted_arrivals:
        flight.assigned_slot_time = None  # Will store assigned slot time in minutes
        flight.assigned_delay = 0  # Will store total delay in minutes
        flight.original_eta_minutes = time_to_minutes(flight.arr_time)
        
        # Flights with delay_type "None" keep their original schedule
        if flight.delay_type == "None":
            flight.assigned_slot_time = flight.original_eta_minutes
            flight.assigned_delay = 0

    # Determine per-hour available slot counts from slots (to enforce capacity)
    # hour -> number of slots available in that hour
    slot_hours = [int(s[0]) // 60 for s in available_slots]
    slot_count_by_hour = Counter(slot_hours)
    # Track assigned count per hour while assigning
    assigned_count_by_hour: dict[int, int] = {}
    # Pre-count flights that keep original schedule (delay_type == 'None') as occupying their hour
    for f in slotted_arrivals:
        if f.delay_type == 'None' and f.assigned_slot_time is not None:
            h = int(f.assigned_slot_time) // 60
            assigned_count_by_hour[h] = assigned_count_by_hour.get(h, 0) + 1
    
    # Sort only the flights that need slots by original ETA for sequential assignment
    flights_needing_slots.sort(key=lambda f: f.original_eta_minutes)
    
    # Phase 1: Assign exempt flights that need Air delay
    exempt_flights_needing_slots = [f for f in flights_needing_slots if f.delay_type == "Air"]
    controlled_flights_needing_slots = [f for f in flights_needing_slots if f.delay_type == "Ground"]
    
    print(f"Assigning slots for {len(exempt_flights_needing_slots)} exempt flights (Air delay) and {len(controlled_flights_needing_slots)} controlled flights (Ground delay)")
    
    def assign_flight_to_slot(flight: Flight, is_exempt: bool) -> bool:
        """
        Assign a flight to the first available slot that's not before its original ETA
        Returns True if assignment successful, False otherwise
        """
        original_eta = flight.original_eta_minutes
        
        # Find the first available slot that's not before the original ETA
        for i, slot in enumerate(available_slots):
            slot_time = slot[0]  # slot time in minutes
            flight_id = slot[1]  # 0 means available
            slot_hour = int(slot_time) // 60

            # Check per-hour capacity: don't assign if this hour is already full
            if assigned_count_by_hour.get(slot_hour, 0) >= slot_count_by_hour.get(slot_hour, 0):
                continue

            # Check if slot is available and not before original ETA
            if flight_id == 0 and slot_time >= original_eta:
                # Assign the flight to this slot
                available_slots[i][1] = hash(flight.callsign) % 10000  # Use hash of callsign as flight ID
                available_slots[i][2] = hash(flight.callsign[:3]) % 1000  # Use airline code hash as airline ID
                
                # Calculate delay
                delay_minutes = slot_time - original_eta
                flight.assigned_slot_time = slot_time
                flight.assigned_delay = delay_minutes
                # increment assigned count for the hour
                assigned_count_by_hour[slot_hour] = assigned_count_by_hour.get(slot_hour, 0) + 1
                flight.assigned_slot_index = i
                
                # Assign delay type based on exemption status
                if is_exempt:
                    available_slots[i][3] = delay_minutes  # Air delay
                    available_slots[i][4] = 0  # No ground delay
                else:
                    available_slots[i][3] = 0  # No air delay
                    available_slots[i][4] = delay_minutes  # Ground delay
                
                return True
        
        return False
    
    # Assign exempt flights first (those needing Air delay)
    assigned_exempt = 0
    for flight in exempt_flights_needing_slots:
        if assign_flight_to_slot(flight, is_exempt=True):
            assigned_exempt += 1
        else:
            print(f"Warning: Could not assign slot to exempt flight {flight.callsign}")
    
    # Assign controlled flights second (those needing Ground delay)
    assigned_controlled = 0
    for flight in controlled_flights_needing_slots:
        if assign_flight_to_slot(flight, is_exempt=False):
            assigned_controlled += 1
        else:
            print(f"Warning: Could not assign slot to controlled flight {flight.callsign}")
    
    # Update the original slots matrix
    slots[:] = available_slots[:]
    
    # Print assignment summary
    print("Slot assignment completed:")
    print(f"  Exempt flights assigned: {assigned_exempt}/{len(exempt_flights_needing_slots)}")
    print(f"  Controlled flights assigned: {assigned_controlled}/{len(controlled_flights_needing_slots)}")
    print(f"  Total flights needing slots: {len(flights_needing_slots)}")
    print(f"  Total flights assigned: {assigned_exempt + assigned_controlled}/{len(flights_needing_slots)}")
    print(f"  Flights keeping original schedule: {len(flights_not_needing_slots)}")
    
    # Calculate delay statistics
    total_air_delay = sum(flight.assigned_delay for flight in exempt_flights_needing_slots if hasattr(flight, 'assigned_delay'))
    total_ground_delay = sum(flight.assigned_delay for flight in controlled_flights_needing_slots if hasattr(flight, 'assigned_delay'))
    
    print(f"  Total air delay: {total_air_delay} minutes")
    print(f"  Total ground delay: {total_ground_delay} minutes")
    print(f"  Average air delay: {total_air_delay/len(exempt_flights_needing_slots):.1f} minutes" if exempt_flights_needing_slots else "  No exempt flights needing slots")
    print(f"  Average ground delay: {total_ground_delay/len(controlled_flights_needing_slots):.1f} minutes" if controlled_flights_needing_slots else "  No controlled flights needing slots")
    # Compute standard deviation and relative standard deviation (RSD)
    # Use sample standard deviation (ddof=1) when there are >=2 samples, otherwise std = 0
    air_delays_list = [flight.assigned_delay for flight in exempt_flights_needing_slots if hasattr(flight, 'assigned_delay')]
    ground_delays_list = [flight.assigned_delay for flight in controlled_flights_needing_slots if hasattr(flight, 'assigned_delay')]

    def stats_with_rsd(values: list[int]) -> tuple[float, float, float]:
        """Return (count, std, rsd_percent). std uses sample std when possible."""
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=float)
        count = len(arr)
        mean = arr.mean()
        std = float(np.std(arr, ddof=1)) if count > 1 else 0.0
        rsd = (std / mean * 100.0) if mean != 0 else 0.0
        return count, std, rsd

    air_count, air_std, air_rsd = stats_with_rsd(air_delays_list)
    ground_count, ground_std, ground_rsd = stats_with_rsd(ground_delays_list)

    if air_count:
        print(f"  Air delays: N={air_count}, Std={air_std:.1f} min, RSD={air_rsd:.1f}%")
    if ground_count:
        print(f"  Ground delays: N={ground_count}, Std={ground_std:.1f} min, RSD={ground_rsd:.1f}%")

    
    return slotted_arrivals


def enforce_capacity(slots: np.ndarray, slotted_arrivals: list[Flight], HStart: int, HEnd: int,
                     reduced_capacity: int, max_capacity: int) -> None:
    """
    Ensure assigned slots per hour do not exceed allowed capacity.
    If an hour is over capacity, move flights (largest delays first) to the next
    available slots after their original ETA. This modifies `slots` and
    `slotted_arrivals` in-place.
    """
    # Build allowed capacity map per hour
    allowed = {}
    for h in range(0, 24):
        if HStart <= h < HEnd:
            allowed[h] = reduced_capacity
        else:
            allowed[h] = max_capacity

    # Map hour -> list of flights assigned in that hour
    hour_map = {}
    for f in slotted_arrivals:
        if getattr(f, 'assigned_slot_time', None) is None:
            continue
        h = int(f.assigned_slot_time // 60)
        hour_map.setdefault(h, []).append(f)

    moved = 0
    # Iterate hours in regulation window first, then others
    hours_to_check = sorted(hour_map.keys())
    for h in hours_to_check:
        current = hour_map.get(h, [])
        cap = allowed.get(h, max_capacity)
        if len(current) <= cap:
            continue
        excess = len(current) - cap
        # Sort flights by assigned_delay descending (move largest delays first)
        candidates = sorted(current, key=lambda x: getattr(x, 'assigned_delay', 0), reverse=True)
        for f in candidates:
            if excess <= 0:
                break
            cur_idx = getattr(f, 'assigned_slot_index', None)
            original_eta = f.arr_time.hour * 60 + f.arr_time.minute
            # Search for next available slot with time >= original_eta
            found = False
            for i in range(cur_idx + 1 if cur_idx is not None else 0, len(slots)):
                if slots[i][1] == 0 and slots[i][0] >= original_eta:
                    # Free old slot
                    if cur_idx is not None and 0 <= cur_idx < len(slots):
                        slots[cur_idx][1] = 0
                        slots[cur_idx][2] = 0
                        slots[cur_idx][3] = 0
                        slots[cur_idx][4] = 0
                    # Assign new slot
                    slots[i][1] = hash(f.callsign) % 10000
                    slots[i][2] = hash(f.callsign[:3]) % 1000
                    delay_minutes = slots[i][0] - original_eta
                    f.assigned_slot_time = int(slots[i][0])
                    f.assigned_slot_index = i
                    f.assigned_delay = int(delay_minutes)
                    if getattr(f, 'is_exempt', False):
                        slots[i][3] = int(delay_minutes)
                        slots[i][4] = 0
                    else:
                        slots[i][3] = 0
                        slots[i][4] = int(delay_minutes)
                    moved += 1
                    excess -= 1
                    found = True
                    break
            if not found:
                # couldn't move this flight — try next candidate
                continue

    if moved:
        print(f"Enforce capacity: moved {moved} flights to respect hourly capacity limits")
    else:
        print("Enforce capacity: no moves required; hourly capacity respected")


def cancel_and_reslot(slotted_arrivals: list[Flight], slots: np.ndarray, company: str = 'VLG', n_cancel: int = 10) -> list[Flight]:
    """
    Cancel top-n highest-delay flights from `company`, then re-run slot assignment giving priority
    to flights from that company to fill the freed slots. Returns the new slotted_arrivals list.

    Behaviour: after cancellation we rebuild available slots (clearing assignments) and
    reassign all flights that require slots (delay_type in (Air, Ground)), prioritizing
    flights whose callsign starts with `company`.

    Returns: new_slotted_arrivals (list[Flight]) with updated assigned_slot_time/assigned_delay
    """

    # Compute metrics helper
    def compute_delay_metrics(flights: list[Flight]):
        air = [getattr(f, 'assigned_delay', 0) for f in flights if f.delay_type == 'Air' and getattr(f, 'assigned_slot_time', None) is not None]
        ground = [getattr(f, 'assigned_delay', 0) for f in flights if f.delay_type == 'Ground' and getattr(f, 'assigned_slot_time', None) is not None]
        def summ(v):
            arr = np.array(v, dtype=float) if v else np.array([], dtype=float)
            return {
                'count': len(arr),
                'total': float(arr.sum()) if arr.size else 0.0,
                'mean': float(arr.mean()) if arr.size else 0.0,
                'std': float(arr.std(ddof=1)) if arr.size>1 else 0.0,
                'rsd': (float(arr.std(ddof=1)) / float(arr.mean()) * 100.0) if arr.size>1 and float(arr.mean()) != 0 else 0.0
            }
        return {'air': summ(air), 'ground': summ(ground)}

    # Snapshot metrics before
    before_metrics = compute_delay_metrics(slotted_arrivals)

    # Identify company flights with assigned delays
    company_flights = [f for f in slotted_arrivals if f.callsign.startswith(company) and getattr(f, 'assigned_slot_time', None) is not None]
    # Sort by assigned_delay descending and pick top n_cancel
    company_flights_sorted = sorted(company_flights, key=lambda f: getattr(f, 'assigned_delay', 0), reverse=True)
    to_cancel = company_flights_sorted[:n_cancel]

    print(f"Cancelling up to {n_cancel} flights from {company} (found {len(to_cancel)})")

    # Mark cancellations and free their slots; record freed indices
    freed_slot_indices = []
    for f in to_cancel:
        idx = getattr(f, 'assigned_slot_index', None)
        if idx is not None and 0 <= idx < len(slots):
            # free the exact slot (leave other slots intact)
            slots[idx][1] = 0
            slots[idx][2] = 0
            slots[idx][3] = 0
            slots[idx][4] = 0
            freed_slot_indices.append(idx)
        f.assigned_slot_time = None
        f.assigned_slot_index = None
        f.assigned_delay = 0
        f.is_cancelled = True

    # Prepare list of flights to reassign: all flights that need slots and are not cancelled
    flights_to_slot = [f for f in slotted_arrivals if f.delay_type in ("Air", "Ground") and not getattr(f, 'is_cancelled', False)]

    # Note: do NOT clear all slots. We only freed the cancelled slots above

    # Ensure flights that had delay_type == 'None' keep their original ETA assigned
    for f in slotted_arrivals:
        if f.delay_type == 'None':
            f.assigned_slot_time = f.arr_time.hour * 60 + f.arr_time.minute
            f.assigned_delay = 0

    # Determine per-hour available slot counts from slots (to enforce capacity during re-assignment)
    slot_hours = [int(s[0]) // 60 for s in slots]
    slot_count_by_hour = Counter(slot_hours)
    # Count currently occupied slots per hour (after cancellations)
    assigned_count_by_hour: dict[int, int] = {}
    for s in slots:
        if s[1] != 0:
            h = int(s[0]) // 60
            assigned_count_by_hour[h] = assigned_count_by_hour.get(h, 0) + 1

    # Prepare candidate groups: company flights first, then others. Within groups, sort by original ETA ascending
    def orig_eta(f):
        return f.arr_time.hour * 60 + f.arr_time.minute

    priority_group = [f for f in flights_to_slot if f.callsign.startswith(company)]
    other_group = [f for f in flights_to_slot if not f.callsign.startswith(company)]
    priority_group.sort(key=orig_eta)
    other_group.sort(key=orig_eta)
    new_queue = priority_group + other_group

    # Iteratively fill freed slots: for each freed slot try to place a company flight whose ETA allows, else put the next flight.
    # When moving an already-assigned flight into a freed slot, its previous slot becomes freed and gets appended to the work queue.
    from collections import deque
    freed_queue = deque(sorted(set(freed_slot_indices)))

    # Helper to find a candidate flight from a list for a given slot_time
    def find_candidate_for_slot(slot_time: int, candidates: list[Flight]):
        # Prefer the earliest original ETA candidate that can be placed in this slot (original_eta <= slot_time)
        for cf in candidates:
            original_eta = cf.arr_time.hour * 60 + cf.arr_time.minute
            # Candidate must be allowed (slot_time >= original ETA)
            if original_eta <= slot_time:
                # Also avoid assigning a flight to a slot earlier than it's already assigned
                if getattr(cf, 'assigned_slot_time', None) is None or (cf.assigned_slot_time is not None and cf.assigned_slot_time > slot_time):
                    return cf
        return None

    # To avoid pathological infinite loops, cap iterations by number of slots * 3
    safety_limit = len(slots) * 3
    iterations = 0

    while freed_queue and iterations < safety_limit:
        iterations += 1
        slot_idx = freed_queue.popleft()
        slot_time = slots[slot_idx][0]
        slot_hour = int(slot_time) // 60

        # Skip if capacity for this hour is already full (shouldn't happen often)
        if assigned_count_by_hour.get(slot_hour, 0) >= slot_count_by_hour.get(slot_hour, 0):
            # cannot place in this hour at the moment; leave it free
            continue

        # Try to find a company flight suitable for this exact slot
        candidate = find_candidate_for_slot(slot_time, priority_group)
        moved = False

        # If no company flight found, try general queue
        if candidate is None:
            candidate = find_candidate_for_slot(slot_time, other_group)

        if candidate is not None:
            # If candidate currently has a slot, free it (it will be appended to the freed_queue)
            old_idx = getattr(candidate, 'assigned_slot_index', None)
            if old_idx is not None and 0 <= old_idx < len(slots):
                # free previous slot
                slots[old_idx][1] = 0
                slots[old_idx][2] = 0
                slots[old_idx][3] = 0
                slots[old_idx][4] = 0
                old_hour = int(slots[old_idx][0]) // 60
                # decrement occupancy from old hour
                assigned_count_by_hour[old_hour] = max(0, assigned_count_by_hour.get(old_hour, 0) - 1)
                # the freed old slot should be considered for reassignment
                freed_queue.append(old_idx)

            # Assign candidate to this slot
            slots[slot_idx][1] = hash(candidate.callsign) % 10000
            slots[slot_idx][2] = hash(candidate.callsign[:3]) % 1000
            delay_minutes = slot_time - (candidate.arr_time.hour * 60 + candidate.arr_time.minute)
            candidate.assigned_slot_time = slot_time
            candidate.assigned_slot_index = slot_idx
            candidate.assigned_delay = delay_minutes

            # increment occupancy for this slot's hour
            assigned_count_by_hour[slot_hour] = assigned_count_by_hour.get(slot_hour, 0) + 1

            # fill the slot delay fields
            if getattr(candidate, 'is_exempt', False):
                slots[slot_idx][3] = delay_minutes
                slots[slot_idx][4] = 0
            else:
                slots[slot_idx][3] = 0
                slots[slot_idx][4] = delay_minutes

            moved = True

        # If we couldn't place any flight into this freed slot, leave it empty and continue
        if not moved:
            # no candidate available for this exact slot time; continue to next freed slot
            continue

    # Compute after metrics
    after_metrics = compute_delay_metrics(slotted_arrivals)

    # Print comparison
    def print_metric_comp(name, before, after):
        print(
            f"{name} - Before: total={before['total']:.1f} min, mean={before['mean']:.1f} min, std={before.get('std',0):.1f} min, RSD={before.get('rsd',0):.1f}%, N={before['count']}; "
            f"After: total={after['total']:.1f} min, mean={after['mean']:.1f} min, std={after.get('std',0):.1f} min, RSD={after.get('rsd',0):.1f}%, N={after['count']}"
        )

    print("\nMETRICS COMPARISON (Before vs After cancellations):")
    print_metric_comp('Air delay', before_metrics['air'], after_metrics['air'])
    print_metric_comp('Ground delay', before_metrics['ground'], after_metrics['ground'])

    # Remove cancelled flights from the returned list so callers treat them as deleted
    cancelled_count = sum(1 for f in slotted_arrivals if getattr(f, 'is_cancelled', False))
    if cancelled_count:
        print(f"Cancelled {cancelled_count} flights from {company} and removed them from the flight list")

    new_slotted_arrivals = [f for f in slotted_arrivals if not getattr(f, 'is_cancelled', False)]

    return new_slotted_arrivals


def print_delay_statistics(slotted_flights: list[Flight]) -> None:
    """
    Print comprehensive delay statistics for different flight categories.
    
    Parameters:
    slotted_flights: list[Flight] - Flights with assigned slots and delays
    """

    # Separate flights by delay type
    air_delay_flights = [f for f in slotted_flights if f.delay_type == "Air" and hasattr(f, 'assigned_delay')]
    ground_delay_flights = [f for f in slotted_flights if f.delay_type == "Ground" and hasattr(f, 'assigned_delay')]
    no_delay_flights = [f for f in slotted_flights if f.delay_type == "None"]
    
    # Extract delay values (in minutes)
    air_delays = [f.assigned_delay for f in air_delay_flights]
    ground_delays = [f.assigned_delay for f in ground_delay_flights]
    no_delays = [0 for _ in no_delay_flights]  # No delay flights have 0 delay
    
    # All delays combined
    all_delays = air_delays + ground_delays + no_delays
    
    def calculate_stats(delays):
        """Calculate and return statistics for a delay category"""
        if not delays:
            return {
                'count': 0,
                'total': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'min': 0,
                'max': 0
            }
        
        delays_array = np.array(delays)
        # Use sample std (ddof=1) when there are multiple samples, otherwise 0
        count = len(delays_array)
        mean = float(np.mean(delays_array))
        std = float(np.std(delays_array, ddof=1)) if count > 1 else 0.0
        rsd = (std / mean * 100.0) if mean != 0 else 0.0
        return {
            'count': count,
            'total': float(np.sum(delays_array)),
            'mean': mean,
            'median': float(np.median(delays_array)),
            'std': std,
            'rsd': rsd,
            'min': float(np.min(delays_array)),
            'max': float(np.max(delays_array))
        }
    
    # Calculate statistics for each category
    air_stats = calculate_stats(air_delays)
    ground_stats = calculate_stats(ground_delays)
    no_delay_stats = calculate_stats(no_delays)
    all_stats = calculate_stats(all_delays)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DELAY STATISTICS")
    print("="*80)
    
    def print_category_stats(stats, category_name):
        """Print statistics for a specific category"""
        print(f"\n{category_name.upper()}:")
        print(f"  Count: {stats['count']} flights")
        print(f"  Total delay: {stats['total']:.0f} minutes ({stats['total']/60:.1f} hours)")
        print(f"  Mean delay: {stats['mean']:.1f} minutes")
        print(f"  Median delay: {stats['median']:.1f} minutes")
        print(f"  Standard deviation: {stats['std']:.1f} minutes")
        print(f"  Relative SD (RSD): {stats.get('rsd', 0.0):.1f}%")
        print(f"  Minimum delay: {stats['min']:.0f} minutes")
        print(f"  Maximum delay: {stats['max']:.0f} minutes")
    
    # Print statistics for each category
    print_category_stats(air_stats, "Air Delay Flights")
    print_category_stats(ground_stats, "Ground Delay Flights")
    print_category_stats(no_delay_stats, "No Delay Flights")
    print_category_stats(all_stats, "All Flights Combined")
    
    # Additional summary
    print(f"\n" + "-"*50)
    print("SUMMARY:")
    print(f"  Total flights processed: {len(slotted_flights)}")
    print(f"  Flights with air delay: {air_stats['count']} ({air_stats['count']/len(slotted_flights)*100:.1f}%)")
    print(f"  Flights with ground delay: {ground_stats['count']} ({ground_stats['count']/len(slotted_flights)*100:.1f}%)")
    print(f"  Flights with no delay: {no_delay_stats['count']} ({no_delay_stats['count']/len(slotted_flights)*100:.1f}%)")
    print(f"  Total system delay: {all_stats['total']:.0f} minutes ({all_stats['total']/60:.1f} hours)")
    print(f"  Average delay per flight: {all_stats['mean']:.1f} minutes")
    
    # Delay efficiency metrics
    regulation_flights = air_stats['count'] + ground_stats['count']
    if regulation_flights > 0:
        avg_regulation_delay = (air_stats['total'] + ground_stats['total']) / regulation_flights
        print(f"  Average delay for regulated flights: {avg_regulation_delay:.1f} minutes")
    
    print("="*80)


# FUNCTION TO COMPUTE RF VECTOR FOR GHP (RF = 1, RF = EMISSIONS, RF = COSTS)
def compute_r_f(flights: list[Flight], objective: str, slot_no: int, flight_no: int,
                slot_times: list[int]) -> np.ndarray:

    penalty_const = 1e12  # penalty definition
    index_count = slot_no * flight_no  # number of possible combination for slot assignment
    r_f = np.zeros(index_count)  # first we set rf vector as all zeros
    cost_arr = np.ones(index_count)

    index = 0
    air_costs = pd.read_csv("Data/AirCosts.csv")  # call the table for air delay costs
    ground_no_reac_costs = pd.read_csv("Data/Ground without reactionary costs.csv")  # call the table for ground not reactionary delay
    ground_costs = pd.read_csv("Data/Ground with reactionary costs.csv")  # call the table for ground reactionary delay
    flight_data = pd.read_csv("Data/LEBL_10AUG2025_ECAC.csv", delimiter=";", encoding='latin-1')

    # for each flight we iterate through the matrix of possible combinations
    for i in range(flight_no):
        flight = flights[i]  # Get the current flight
        original_arrival = flight.arr_time.hour * 60 + flight.arr_time.minute  # original ETA to minutes

        for j in range(slot_no):
            delay = slot_times[j] - original_arrival  # delay = CTA (slot time) - ETA => CTA = ETA + delay

            # if negative delay -> slot assigned BEFORE flight arrival -> IMPOSSIBLE -> high penalty
            if delay < 0:
                delay = penalty_const

            # Apply objective-specific costs
            match objective:
                case "emissions":  # GHP computation minimizing CO2 EMISSIONS
                    if flight.delay_type.upper() == "AIR":  # air delay
                        emissions = flight.compute_air_del_emissions(delay, "delay") * delay # call compute air del emissions
                    elif flight.delay_type.upper() == "GROUND":  # ground delay
                        emissions = flight.compute_ground_del_emissions(delay)  # call compute ground del emissions
                    else:  # non delay -> no cost associated to the delay
                        emissions = 1
                    r_f[index] = emissions

                case "costs":  # GHP computation minimizing ECONOMIC COSTS for operators
                    cost = flight.compute_costs(delay, air_costs, ground_no_reac_costs, ground_costs, flight_data)  # inputs for the functions: delay and tables
                    r_f[index] = cost  # value of the economic cost that the delay, obtained by assign the flight f at slot t, generates

                case "delay":  # Default case
                    r_f[index] = delay  # 1*delay

            index += 1

    return r_f





def compute_GHP(filtered_arrivals: list[Flight], slots: np.ndarray, objective = Literal["delay", "emissions", "costs"]):
    """
    Solve GHP as an integer program:
      - filtered_arrivals: list of Flight objects (they must have .delay_type, .arr_time, .seats, etc.)
      - slots: numpy array with first column slot_time (in minutes)
      - rf_vector: optional list with one rf per flight (len = number of flights needing slots).
                   If None and objective == 'emissions', rf computed from flight emissions per minute
                   If None and objective == 'delay', rf defaults to 1 for all flights (validation).
      - objective: 'delay' or 'emissions'
    Returns: list of flights with assigned_slot_time and assigned_delay updated (same convention que assignSlotsGDP)
    """

    # Select flights that need slot assignment (Air or Ground)
    flights_needing_slots = [f for f in filtered_arrivals if f.delay_type in ("Air", "Ground")]

    # number of flights -> number of rows of the matrix
    n_f = len(flights_needing_slots)
    if n_f == 0:
        print("No flights require regulation. Nothing to solve.")
        return filtered_arrivals

    # number of slots -> number of columns of the matrix
    slot_times = [int(s[0]) for s in slots]  # minutes
    n_s = len(slot_times)

    #computation of rf vector (ALREADY COMPUTED AS A COST VECTOR)
    r_f = compute_r_f(flights_needing_slots, str(objective), n_s, n_f, slot_times)

    # Number of variables: n_f * n_s
    n_vars = n_f * n_s

    # --------------------------------------------------------
    # CONSTRAINT 1: Each flight assigned to exactly 1 slot (summatory of possible slots for a unique flight is 1)
    # EQUALITY CONSTRAINT
    a_eq = np.zeros((n_f, n_vars))
    for i in range(n_f):
        a_eq[i, i * n_s:(i + 1) * n_s] = 1
    b_eq = np.ones(n_f)

    # CONSTRAINT 2: Each slot capacity not exceeded (summatory of possible flights assigned to a unique slot <= 1, it could be assigned or not)
    # INEQUALITY CONSTRAINT
    a_ub = np.zeros((n_s, n_vars))
    for j in range(n_s):
        a_ub[j, j::n_s] = 1
    b_ub = np.ones(n_s)

    # Solve as integer program thanks to SCIPY LIBRARY
    result = linprog(r_f, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=(0, 1), method='highs', integrality=1)  # INTEGER VARIABLES: integrality = 1, Binary variables: ub = 1, lb = 0

    if result.success:
        print("Optimization successful!")

        # Print assignment results with delays
        print("\nASSIGNMENTS & DELAYS:")
        print("=" * 60)
        total_air_delay = 0  # accumulative total air delay summarying each particular air delay
        total_ground_delay = 0  # accumulative total ground delay summarying each particular ground delay

        for i in range(n_f):
            for j in range(n_s):
                var_index = i * n_s + j
                if abs(result.x[var_index] - 1) < 1e-6:  # if found that a flight f assigned to slot t (1-1 just to avoid problems)
                    flight = flights_needing_slots[i]  # flight designated with del type "ground" or "air".

                    callsign = getattr(flight, 'callsign', f'Flight{i + 1}')  # get callsign for possible reactionary delay

                    # Calculate delays
                    original_arrival = flight.arr_time.hour * 60 + flight.arr_time.minute  # ETA to minutes
                    slot_time = slot_times[j]
                    total_delay = max(0, slot_time - original_arrival) # -> DELAY = CTA (slot time) - ETA (original arrival time)

                    # Split delay based on delay type
                    if flight.delay_type == "Air":
                        air_delay = total_delay  # all is air delay
                        ground_delay = 0  # not ground delay
                    else:  # "Ground"
                        air_delay = 0  # not air delay
                        ground_delay = total_delay  # all is ground delay

                    # Update totals
                    total_air_delay += air_delay
                    total_ground_delay += ground_delay

                    print(f"{callsign} → Slot {j + 1} | "
                          f"Air: {air_delay:3d}min | "
                          f"Ground: {ground_delay:3d}min | "
                          f"Total: {total_delay:3d}min")

        # Print summary
        print("=" * 60)
        print(f"TOTALS: "
              f"Air: {total_air_delay:3d}min | "
              f"Ground: {total_ground_delay:3d}min | "
              f"Total: {total_air_delay + total_ground_delay:3d}min")

    else:
        print("Optimization failed:", result.message)

    return filtered_arrivals  # return of the updated list. #TODO



# FUNCTION THAT COMPUTE THE RAIL EMISSIONS OF THE FLIGHTS THAT COULD yes BE REPLACED BY A LESS THAN 6 HOUR DIRECT TRAIN ROUTE FROM BCN
def compute_Rail_Emissions_D2DTime(filtered_arrivals: list['Flight']):
    airports = ["LEMD", "LFML", "LEZL", "LEMG", "LFLL", "LEAL"] #LIST OF ARIPORTS OF CITIES THAT HAVE A POSSIBLE DIRECT TRAIN ROUTE FROM BARCELONA.

    rail_trips = [f for f in filtered_arrivals if f.departure_airport in airports] #OBTAIN NEW FLIGHT LIST WITHOUT FLIGHTS THAT COULD BE DONE BY TRAIN

    rail_trip_time = [99, 310, 376, 396, 301, 344] #We later sum 60' for the D2D time
    rail_emissions = [17.4, 7.2, 31.8, 32.5, 7.9, 15.5] #emissions in [kg CO2/train journey].

    total_rail_emissions = 0
    D2D_rail_time = 0

    for f in rail_trips: #loop inside the replaceable flights to search the duration of the journey by train and its emissions.
        for i in range(len(airports)):
            if f.departure_airport == airports[i]:
                total_rail_emissions += rail_emissions[i] # look for the emissions that correspond to the route and add it to the global computation
                D2D_rail_time += (rail_trip_time[i] + 60) # look for the time that correspond to the route and add it to the total.
    return rail_trips, total_rail_emissions, D2D_rail_time

# FUNCTION THAT COMPUTE THE RAIL EMISSIONS OF THE FLIGHTS THAT COULD not BE REPLACED BY A LESS THAN 6 HOUR DIRECT TRAIN ROUTE FROM BCN
def compute_Flight_Emissions_D2DTime(filtered_arrivals: list['Flight'], delay: int):
    airports = ["LEMD", "LFML", "LEZL", "LEMG", "LFLL", "LEAL"]
    flight_trips = [f for f in filtered_arrivals if f.departure_airport not in airports] #LIST OF ARIPORTS OF CITIES THAT do not HAVE A POSSIBLE DIRECT TRAIN ROUTE FROM BARCELONA.
    flight_trip_time = [164, 153, 178, 181, 183, 161] #We later sum 150' for the D2D time
    flight_emissions = [115.41, 128.39, 146.94, 139.54, 116.27, 101.1] #emissions in [kg CO2/flight journey].

    total_flight_emissions = 0
    D2D_aircraft_time = 0

    for f in flight_trips: #loop inside the replaceable flights to search the duration of the journey by train and its emissions.
        for i in range(len(airports)):
            if f.departure_airport == airports[i]:
                total_flight_emissions += flight_emissions[i] # look for the emissions that correspond to the route and add it to the global computation
                D2D_aircraft_time += (flight_trip_time[i] + 150 + delay) # look for the time that correspond to the route and add it to the total.
    return flight_trips, total_flight_emissions, D2D_aircraft_time

def plot_train_vs_flight_emissions_times():
    # Datos base
    airports = ["LEGE", "LEMD", "LFML", "LEZL", "LEMG"]
    rail_emissions = [2.9, 17.4, 7.2, 31.8, 32.5]
    flight_emissions = [..., 115.41, 128.39, 146.94, 139.54]
    rail_time = [38 + 60, 99 + 60, 310 + 60, 376 + 60, 396 + 60]
    flight_time = [150, 164 + 150, 153 + 150, 178 + 150, 181 + 150]

    # Posiciones en el eje X
    x = np.arange(len(airports))
    width = 0.35

    # Crear figura
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Barras de emisiones
    ax1.bar(x - width/2, rail_emissions, width, label='Rail CO₂ (kg)')
    ax1.bar(x + width/2, flight_emissions, width, label='Flight CO₂ (kg)')
    ax1.set_ylabel('CO₂ emissions (kg)')
    ax1.set_xlabel('Departure Airport')
    ax1.set_title('Comparación: emisiones y tiempos puerta a puerta (Tren vs Avión)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(airports)
    ax1.legend(loc='upper left')

    # Eje secundario para tiempos
    ax2 = ax1.twinx()
    ax2.plot(x, rail_time, marker='o', color='green', label='Rail time (min)')
    ax2.plot(x, flight_time, marker='o', color='red', label='Flight time (min)')
    ax2.set_ylabel('Door-to-door time (min)')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()



def plot_hfile_analysis(arrival_flights: list[Flight], distThreshold: int, HStart: int,
                        HEnd: int, reduced_capacity: int, max_capacity: int,
                        verbose: bool = False) -> None:
    """
    Plot Air Delay, Ground Delay, Unrecoverable Delay, and Emissions against HFile values.
    
    Parameters:
    arrival_flights: list[Flight] - All arrival flights
    distThreshold: int - Distance threshold in km
    HStart: int - Regulation start hour
    HEnd: int - Regulation end hour
    reduced_capacity: int - Reduced capacity during regulation
    max_capacity: int - Maximum capacity
    """
    
    # HFile values from 6h up to (but not including) 11h in 10-minute increments
    # Stop before regulation start at 11:00 so HFile < HStart
    hfile_values = []
    for hour in range(6, 11):  # 6 to 10 hours (last HFile = 10:50)
        for minute in [0, 10, 20, 30, 40, 50]:
            hfile_values.append(hour + minute/60)
    
    # Initialize lists to store metrics
    air_delays = []
    ground_delays = []
    unrecoverable_delays = []
    total_emissions = []
    
    if verbose:
        print("\n" + "="*80)
        print("ANALYZING HFILE VARIATIONS (6:00 to 11:00)")
        print("="*80)
    
    # For each HFile value, compute the metrics
    for HFile in hfile_values:
        # Compute HNoReg properly for this HFile value
        HNoReg = compute_hnoreg(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)
        
        # Filter flights for this HFile
        filtered_flights = filter_arrival_flights(arrival_flights, distThreshold, HStart, HNoReg, HFile)
        
        # Compute slots
        Hstart_min = HStart * 60
        Hend_min = HEnd * 60
        HNoReg_min = HNoReg * 60
        extended_HNoReg = HNoReg_min + 30
        slots = compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)
        
        # Assign slots
        slotted_arrivals = assignSlotsGDP(filtered_flights, slots)
        
        # Calculate metrics
        total_air_delay = 0
        total_ground_delay = 0
        total_unrecoverable_delay = 0
        air_emissions = 0
        ground_emissions = 0
        
        for flight in slotted_arrivals:
            delay = getattr(flight, 'assigned_delay', 0)
            delay_type = getattr(flight, 'delay_type', 'None')
            
            # Calculate unrecoverable delay
            total_unrecoverable_delay += flight.computeunrecdel(delay, HStart)
            
            # Calculate delays and emissions by type
            if delay_type == "Air":
                total_air_delay += delay
                air_emissions += flight.compute_air_del_emissions(delay, "delay")
            elif delay_type == "Ground":
                total_ground_delay += delay
                if delay > 60:
                    ground_emissions += flight.compute_ground_del_emissions(delay) / 9
                else:
                    ground_emissions += flight.compute_ground_del_emissions(delay)
        
        # Store metrics
        air_delays.append(total_air_delay)
        ground_delays.append(total_ground_delay)
        unrecoverable_delays.append(total_unrecoverable_delay)
        total_emissions.append(air_emissions + ground_emissions)
        
        if verbose:
            print(f"HFile={HFile:.2f}h: Air={total_air_delay:.1f}min, Ground={total_ground_delay:.1f}min, "
                f"Unrec={total_unrecoverable_delay:.1f}min, Emissions={air_emissions + ground_emissions:.2f}kg CO2")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot delays on primary y-axis (left)
    ax1.set_xlabel('HFile (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Delay (minutes)', fontsize=12, fontweight='bold')
    
    line1 = ax1.plot(hfile_values, air_delays, 'o-', color='blue', linewidth=2, 
                     markersize=4, label='Air Delay (min)')
    line2 = ax1.plot(hfile_values, ground_delays, 's-', color='red', linewidth=2, 
                     markersize=4, label='Ground Delay (min)')
    line3 = ax1.plot(hfile_values, unrecoverable_delays, '^-', color='orange', linewidth=2, 
                     markersize=4, label='Unrecoverable Delay (min)')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create secondary y-axis for emissions (right)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Emissions (kg CO2)', fontsize=12, fontweight='bold')
    
    line4 = ax2.plot(hfile_values, total_emissions, 'd-', color='green', linewidth=2, 
                     markersize=4, label='Total Emissions (kg CO2)')
    
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
    
    # Set title
    plt.title('Impact of HFile on Air Delay, Ground Delay, Unrecoverable Delay, and Emissions', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show hours properly (6:00 to 11:00)
    ax1.set_xlim(6, 11)
    ax1.set_xticks([i for i in range(6, 12)])
    ax1.set_xticklabels([f'{i}:00' for i in range(6, 12)])
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        print("="*80)
        print("HFILE ANALYSIS COMPLETE")
        print("="*80)


def plot_distance_threshold_analysis(arrival_flights: list[Flight], HFile: int, HStart: int, 
                                     HEnd: int, reduced_capacity: int, max_capacity: int,
                                     verbose: bool = False) -> None:
    """
    Plot Air Delay, Ground Delay, Unrecoverable Delay, and Emissions against Distance Threshold values.
    
    Parameters:
    arrival_flights: list[Flight] - All arrival flights
    HFile: int - Filing hour
    HStart: int - Regulation start hour
    HEnd: int - Regulation end hour
    reduced_capacity: int - Reduced capacity during regulation
    max_capacity: int - Maximum capacity
    """
    
    # Distance threshold values from 0km to 3000km in 50km increments
    dist_threshold_values = list(range(0, 3001, 50))
    
    # Initialize lists to store metrics
    air_delays = []
    ground_delays = []
    unrecoverable_delays = []
    total_emissions = []
    
    if verbose:
        print("\n" + "="*80)
        print("ANALYZING DISTANCE THRESHOLD VARIATIONS (0 to 3000km)")
        print("="*80)
    
    # For each distance threshold value, compute the metrics
    for distThreshold in dist_threshold_values:
        # Compute HNoReg properly for this distance threshold
        HNoReg = compute_hnoreg(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)
        
        # Filter flights for this distance threshold
        filtered_flights = filter_arrival_flights(arrival_flights, distThreshold, HStart, HNoReg, HFile)
        
        # Compute slots
        Hstart_min = HStart * 60
        Hend_min = HEnd * 60
        HNoReg_min = HNoReg * 60
        extended_HNoReg = HNoReg_min + 30
        slots = compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)
        
        # Assign slots
        slotted_arrivals = assignSlotsGDP(filtered_flights, slots)
        
        # Calculate metrics
        total_air_delay = 0
        total_ground_delay = 0
        total_unrecoverable_delay = 0
        air_emissions = 0
        ground_emissions = 0
        
        for flight in slotted_arrivals:
            delay = getattr(flight, 'assigned_delay', 0)
            delay_type = getattr(flight, 'delay_type', 'None')
            
            # Calculate unrecoverable delay
            total_unrecoverable_delay += flight.computeunrecdel(delay, HStart)
            
            # Calculate delays and emissions by type
            if delay_type == "Air":
                total_air_delay += delay
                air_emissions += flight.compute_air_del_emissions(delay, "delay")
            elif delay_type == "Ground":
                total_ground_delay += delay
                if delay > 60:
                    ground_emissions += flight.compute_ground_del_emissions(delay) / 9
                else:
                    ground_emissions += flight.compute_ground_del_emissions(delay)
        
        # Store metrics
        air_delays.append(total_air_delay)
        ground_delays.append(total_ground_delay)
        unrecoverable_delays.append(total_unrecoverable_delay)
        total_emissions.append(air_emissions + ground_emissions)
        
        # Print progress every 500km (only when verbose)
        if verbose and (distThreshold % 500 == 0 or distThreshold == 200):
            print(f"DistThreshold={distThreshold}km: Air={total_air_delay:.1f}min, Ground={total_ground_delay:.1f}min, "
                  f"Unrec={total_unrecoverable_delay:.1f}min, Emissions={air_emissions + ground_emissions:.2f}kg CO2")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot delays on primary y-axis (left)
    ax1.set_xlabel("Distance Threshold (km)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Delay (minutes)", fontsize=12, fontweight="bold")
    
    line1 = ax1.plot(dist_threshold_values, air_delays, "o-", color="blue", linewidth=2, 
                     markersize=3, label="Air Delay (min)")
    line2 = ax1.plot(dist_threshold_values, ground_delays, "s-", color="red", linewidth=2, 
                     markersize=3, label="Ground Delay (min)")
    line3 = ax1.plot(dist_threshold_values, unrecoverable_delays, "^-", color="orange", linewidth=2, 
                     markersize=3, label="Unrecoverable Delay (min)")
    
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Create secondary y-axis for emissions (right)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Emissions (kg CO2)", fontsize=12, fontweight="bold")
    
    line4 = ax2.plot(dist_threshold_values, total_emissions, "d-", color="green", linewidth=2, 
                     markersize=3, label="Total Emissions (kg CO2)")
    
    ax2.tick_params(axis="y", labelcolor="green")
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=10, framealpha=0.9)
    
    # Set title
    plt.title("Impact of Distance Threshold on Air Delay, Ground Delay, Unrecoverable Delay, and Emissions", 
              fontsize=14, fontweight="bold", pad=20)
    
    # Format x-axis (0 to 3000km)
    ax1.set_xlim(0, 3000)
    
    plt.tight_layout()
    plt.show()
    
    print("="*80)
    print("DISTANCE THRESHOLD ANALYSIS COMPLETE")
    print("="*80)


def plot_3d_analysis(arrival_flights: list[Flight], HStart: int, HEnd: int, 
                     reduced_capacity: int, max_capacity: int) -> None:
    """
    Create 3D surface plots and heatmap for HFile vs Distance Threshold analysis.
    Generates 4 subplots:
    1. 3D surface: Unrecoverable Delay
    2. 3D surface: Air Delay  
    3. 3D surface: Total Emissions
    4. Heatmap: Combined normalized score with optimal point
    
    Parameters:
    arrival_flights: list[Flight] - All arrival flights
    HStart: int - Regulation start hour
    HEnd: int - Regulation end hour
    reduced_capacity: int - Reduced capacity during regulation
    max_capacity: int - Maximum capacity
    """
    import sys
    import io
    
    print("\n" + "="*80)
    print("GENERATING 3D ANALYSIS: HFILE vs DISTANCE THRESHOLD")
    print("="*80)
    
    # Define ranges
    # HFile values from 6h up to (but not including) 11h in 10-minute increments
    # Stop before regulation start at 11:00 so HFile < HStart
    hfile_values = []
    for hour in range(6, 11):  # 6 to 10 hours (last HFile = 10:50)
        for minute in [0, 10, 20, 30, 40, 50]:
            hfile_values.append(hour + minute/60)
    
    dist_threshold_values = list(range(0, 3001, 50))
    
    # Initialize 2D arrays to store results
    unrecoverable_delays = np.zeros((len(hfile_values), len(dist_threshold_values)))
    air_delays = np.zeros((len(hfile_values), len(dist_threshold_values)))
    total_emissions = np.zeros((len(hfile_values), len(dist_threshold_values)))
    
    print(f"Computing {len(hfile_values)} HFile values × {len(dist_threshold_values)} distance thresholds = {len(hfile_values) * len(dist_threshold_values)} iterations...")
    print("Suppressing console output during computation...")
    
    # Compute metrics for each combination
    iteration = 0
    total_iterations = len(hfile_values) * len(dist_threshold_values)
    
    # Suppress stdout during loop to avoid clutter
    original_stdout = sys.stdout
    
    for i, HFile in enumerate(hfile_values):
        for j, distThreshold in enumerate(dist_threshold_values):
            iteration += 1
            
            # Print progress every 100 iterations (restore stdout temporarily)
            if iteration % 100 == 0 or iteration == total_iterations:
                sys.stdout = original_stdout
                print(f"Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")
                sys.stdout = io.StringIO()  # Suppress again
            
            # Suppress console output for these function calls
            if sys.stdout == original_stdout:
                sys.stdout = io.StringIO()
            # Suppress console output for these function calls
            if sys.stdout == original_stdout:
                sys.stdout = io.StringIO()
            
            # Compute HNoReg
            HNoReg = compute_hnoreg(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)
            
            # Filter flights
            filtered_flights = filter_arrival_flights(arrival_flights, distThreshold, HStart, HNoReg, HFile)
            
            # Compute slots
            Hstart_min = HStart * 60
            Hend_min = HEnd * 60
            HNoReg_min = HNoReg * 60
            extended_HNoReg = HNoReg_min + 30
            slots = compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)
            
            # Assign slots
            slotted_arrivals = assignSlotsGDP(filtered_flights, slots)
            
            # Calculate metrics
            total_air_delay = 0
            total_unrecoverable_delay = 0
            air_emissions = 0
            ground_emissions = 0
            
            for flight in slotted_arrivals:
                delay = getattr(flight, 'assigned_delay', 0)
                delay_type = getattr(flight, 'delay_type', 'None')
                
                # Calculate unrecoverable delay
                total_unrecoverable_delay += flight.computeunrecdel(delay, HStart)
                
                # Calculate delays and emissions by type
                if delay_type == "Air":
                    total_air_delay += delay
                    air_emissions += flight.compute_air_del_emissions(delay, "delay")
                elif delay_type == "Ground":
                    if delay > 60:
                        ground_emissions += flight.compute_ground_del_emissions(delay) / 9
                    else:
                        ground_emissions += flight.compute_ground_del_emissions(delay)
            
            # Store in 2D arrays
            unrecoverable_delays[i, j] = total_unrecoverable_delay
            air_delays[i, j] = total_air_delay
            total_emissions[i, j] = air_emissions + ground_emissions
    
    # Restore stdout
    sys.stdout = original_stdout
    print("Computation complete! Creating visualizations...")
    
    # Create meshgrid for 3D plotting
    X, Y = np.meshgrid(dist_threshold_values, hfile_values)
    
    # We'll save each plot as its own high-quality image for better clarity
    
    # Normalize each metric to 0-1 before plotting the surfaces (preserve values for weighted combination)
    norm_unrec = (unrecoverable_delays - unrecoverable_delays.min()) / (unrecoverable_delays.max() - unrecoverable_delays.min() + 1e-10)
    norm_air = (air_delays - air_delays.min()) / (air_delays.max() - air_delays.min() + 1e-10)
    norm_emissions = (total_emissions - total_emissions.min()) / (total_emissions.max() - total_emissions.min() + 1e-10)

    # 1. 3D Surface: Unrecoverable Delay (raw values)
    fig1 = plt.figure(figsize=(16, 12))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf1 = ax1.plot_surface(X, Y, unrecoverable_delays, cmap='viridis', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Distance Threshold (km)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('HFile (hours)', fontsize=10, fontweight='bold')
    ax1.set_zlabel('Unrecoverable Delay (min)', fontsize=10, fontweight='bold')
    ax1.set_title('Unrecoverable Delay vs HFile & Distance Threshold', fontsize=12, fontweight='bold', pad=15)
    # move the colorbar slightly away from the axes to add whitespace (pad controls distance)
    fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, pad=0.14)
    # Improve readability: larger fonts and more whitespace
    ax1.title.set_fontsize(18)
    ax1.xaxis.label.set_size(14)
    ax1.yaxis.label.set_size(14)
    ax1.zaxis.label.set_size(14)
    # add label padding so titles/labels aren't cramped against axes
    ax1.xaxis.labelpad = 12
    ax1.yaxis.labelpad = 12
    try:
        ax1.zaxis.labelpad = 10
    except Exception:
        pass
    ax1.tick_params(axis='both', which='major', labelsize=12)
    if hasattr(ax1, 'zaxis'):
        ax1.zaxis.set_major_formatter(lambda x, pos: f"{x:.0f}")
    fig1.tight_layout(pad=3.0)
    fig1.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)
    plt.show()

    # 2. 3D Surface: Air Delay (raw values)
    # 2. 3D Surface: Air Delay (raw values)
    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(X, Y, air_delays, cmap='plasma', alpha=0.9, edgecolor='none')
    ax2.set_xlabel('Distance Threshold (km)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('HFile (hours)', fontsize=10, fontweight='bold')
    ax2.set_zlabel('Air Delay (min)', fontsize=10, fontweight='bold')
    ax2.set_title('Air Delay vs HFile & Distance Threshold', fontsize=12, fontweight='bold', pad=15)
    fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, pad=0.14)
    ax2.title.set_fontsize(18)
    ax2.xaxis.label.set_size(14)
    ax2.yaxis.label.set_size(14)
    ax2.zaxis.label.set_size(14)
    ax2.xaxis.labelpad = 12
    ax2.yaxis.labelpad = 12
    try:
        ax2.zaxis.labelpad = 10
    except Exception:
        pass
    ax2.tick_params(axis='both', which='major', labelsize=12)
    fig2.tight_layout(pad=3.0)
    fig2.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)
    plt.show()

    # 3. 3D Surface: Total Emissions (raw values)
    # 3. 3D Surface: Total Emissions (raw values)
    fig3 = plt.figure(figsize=(16, 12))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf3 = ax3.plot_surface(X, Y, total_emissions, cmap='inferno', alpha=0.9, edgecolor='none')
    ax3.set_xlabel('Distance Threshold (km)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('HFile (hours)', fontsize=10, fontweight='bold')
    ax3.set_zlabel('Total Emissions (kg CO2)', fontsize=10, fontweight='bold')
    ax3.set_title('Total Emissions vs HFile & Distance Threshold', fontsize=12, fontweight='bold', pad=15)
    fig3.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5, pad=0.14)
    ax3.title.set_fontsize(18)
    ax3.xaxis.label.set_size(14)
    ax3.yaxis.label.set_size(14)
    ax3.zaxis.label.set_size(14)
    ax3.xaxis.labelpad = 12
    ax3.yaxis.labelpad = 12
    try:
        ax3.zaxis.labelpad = 10
    except Exception:
        pass
    ax3.tick_params(axis='both', which='major', labelsize=12)
    fig3.tight_layout(pad=3.0)
    fig3.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)
    plt.show()
    
    # 4. Heatmap: Combined normalized score
    fig4 = plt.figure(figsize=(12, 10))
    ax4 = fig4.add_subplot(111)
    
    # Normalize each metric to 0-1 scale (lower is better)
    norm_unrec = (unrecoverable_delays - unrecoverable_delays.min()) / (unrecoverable_delays.max() - unrecoverable_delays.min() + 1e-10)
    norm_air = (air_delays - air_delays.min()) / (air_delays.max() - air_delays.min() + 1e-10)
    norm_emissions = (total_emissions - total_emissions.min()) / (total_emissions.max() - total_emissions.min() + 1e-10)
    
    # Define weights for each metric (adjust these based on priority)
    # Higher weight = more importance in the final score
    weight_unrec = 10     # Unrecoverable delay weight
    weight_air = 20        # Air delay weight
    weight_emissions = 1   # Emissions weight
    
    # Combined score with weighted normalization
    combined_score = (weight_unrec * norm_unrec + 
                     weight_air * norm_air + 
                     weight_emissions * norm_emissions)
    
    # Normalize combined score to 0-1 for better visualization
    combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min() + 1e-10)
    
    # Find optimal point (minimum combined score)
    min_idx = np.unravel_index(np.argmin(combined_score), combined_score.shape)
    optimal_hfile = hfile_values[min_idx[0]]
    optimal_dist = dist_threshold_values[min_idx[1]]
    
    # Create heatmap
    im = ax4.imshow(combined_score, cmap='RdYlGn_r', aspect='auto', origin='lower',
                    extent=[dist_threshold_values[0], dist_threshold_values[-1],
                           hfile_values[0], hfile_values[-1]])
    
    # Mark optimal point
    ax4.plot(optimal_dist, optimal_hfile, 'b*', markersize=20, markeredgecolor='white', 
             markeredgewidth=2, label=f'Optimal: HFile={optimal_hfile:.2f}h, Dist={optimal_dist}km')
    
    ax4.set_xlabel('Distance Threshold (km)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('HFile (hours)', fontsize=10, fontweight='bold')
    ax4.set_title(f'Combined Weighted Score Heatmap\n(Weights: Unrec={weight_unrec}, Air={weight_air}, Emis={weight_emissions})', 
                  fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = fig4.colorbar(im, ax=ax4)
    cbar.set_label('Combined Score', fontsize=10)
    
    # Add legend
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Add grid
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Improve heatmap readability: larger fonts and more whitespace
    ax4.title.set_fontsize(16)
    ax4.xaxis.label.set_size(14)
    ax4.yaxis.label.set_size(14)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    cbar.ax.tick_params(labelsize=12)
    fig4.tight_layout(pad=3.0)
    fig4.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.08)
    plt.show()
    
    # Print data ranges to understand scales
    print("\n" + "="*80)
    print("DATA RANGES (Raw Values):")
    print("="*80)
    print(f"Unrecoverable Delay: {unrecoverable_delays.min():.1f} - {unrecoverable_delays.max():.1f} minutes (Range: {unrecoverable_delays.max() - unrecoverable_delays.min():.1f})")
    print(f"Air Delay:           {air_delays.min():.1f} - {air_delays.max():.1f} minutes (Range: {air_delays.max() - air_delays.min():.1f})")
    print(f"Total Emissions:     {total_emissions.min():.2f} - {total_emissions.max():.2f} kg CO2 (Range: {total_emissions.max() - total_emissions.min():.2f})")
    print("="*80)
    print(f"\nWeights applied:")
    print(f"  Unrecoverable Delay: {weight_unrec}")
    print(f"  Air Delay: {weight_air}")
    print(f"  Emissions: {weight_emissions}")
    
    # Print optimal values
    print("\n" + "="*80)
    print("OPTIMAL CONFIGURATION:")
    print("="*80)
    print(f"HFile: {optimal_hfile:.2f} hours ({int(optimal_hfile)}:{int((optimal_hfile % 1) * 60):02d})")
    print(f"Distance Threshold: {optimal_dist} km")
    print(f"\nAt this configuration:")
    print(f"  Unrecoverable Delay: {unrecoverable_delays[min_idx[0], min_idx[1]]:.1f} minutes")
    print(f"  Air Delay: {air_delays[min_idx[0], min_idx[1]]:.1f} minutes")
    print(f"  Total Emissions: {total_emissions[min_idx[0], min_idx[1]]:.2f} kg CO2")
    print(f"  Combined Normalized Score: {combined_score[min_idx[0], min_idx[1]]:.4f}")
    print("="*80)


if __name__ == "__main__":
    print("You're not executing the main program")
