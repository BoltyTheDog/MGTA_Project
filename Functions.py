import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from Classes.Flight import Flight
import numpy as np
from collections import Counter
import math
import pulp

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
    
    plt.legend()
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
            
            # Check if slot is available and not before original ETA
            if flight_id == 0 and slot_time >= original_eta:
                # Assign the flight to this slot
                available_slots[i][1] = hash(flight.callsign) % 10000  # Use hash of callsign as flight ID
                available_slots[i][2] = hash(flight.callsign[:3]) % 1000  # Use airline code hash as airline ID
                
                # Calculate delay
                delay_minutes = slot_time - original_eta
                flight.assigned_slot_time = slot_time
                flight.assigned_delay = delay_minutes
                
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
    
    return slotted_arrivals


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
        return {
            'count': len(delays),
            'total': np.sum(delays_array),
            'mean': np.mean(delays_array),
            'median': np.median(delays_array),
            'std': np.std(delays_array),
            'min': np.min(delays_array),
            'max': np.max(delays_array)
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




def compute_GHP(filtered_arrivals: list['Flight'], slots: np.ndarray, rf_vector: list | None = None, objective: str = 'delay'):
    """
    Solve GHP as an integer program:
      - filtered_arrivals: list of Flight objects (they must have .delay_type, .arr_time, .seats, etc)
      - slots: numpy array with first column slot_time (in minutes)
      - rf_vector: optional list with one rf per flight (len = number of flights needing slots).
                   If None and objective == 'emissions', rf computed from flight emissions per minute
                   If None and objective == 'delay', rf defaults to 1 for all flights (validation).
      - objective: 'delay' or 'emissions'
    Returns: list of flights with assigned_slot_time and assigned_delay updated (same convention que assignSlotsGDP)
    """
    # Helper: convert datetime -> minutes since midnight
    def time_to_minutes(time_obj):
        return time_obj.hour * 60 + time_obj.minute

    # Select flights that need slot assignment (Air or Ground)
    flights_needing_slots = [f for f in filtered_arrivals if f.delay_type in ("Air", "Ground")]
    flights_not_needing = [f for f in filtered_arrivals if f.delay_type == "None"]
    n_f = len(flights_needing_slots)
    if n_f == 0:
        print("No flights require regulation. Nothing to solve.")
        return filtered_arrivals

    slot_times = [int(s[0]) for s in slots]  # minutos
    n_s = len(slot_times)

    # Map index
    f_index = {i: flights_needing_slots[i] for i in range(n_f)}
    s_index = {j: slot_times[j] for j in range(n_s)}

    # Build rf per flight (per-minute cost factor)
    rf = [1.0] * n_f  # default to 1 (validation)
    if rf_vector is not None:
        # If provided, expect same order as flights_needing_slots
        if len(rf_vector) != n_f:
            print("Warning: rf_vector length mismatch; ignoring rf_vector and computing from flights.")
        else:
            rf = list(rf_vector)
    elif objective == 'emissions':
        # compute per-flight emissions per minute depending on delay_type
        rf = []
        for f in flights_needing_slots:
            if f.delay_type == "Air":
                # compute_air_del_emissions returns kg CO2/min according to your Flight class docstring
                rf.append(f.compute_air_del_emissions())
            else:  # Ground
                rf.append(f.compute_ground_del_emissions())

    # Build candidate slot-feasible matrix: allow only slots >= ETA
    ETA_minutes = [time_to_minutes(f.arr_time) for f in flights_needing_slots]
    feasible_pairs = []
    for i in range(n_f):
        for j in range(n_s):
            if s_index[j] >= ETA_minutes[i]:
                feasible_pairs.append((i, j))

    # Create pulp problem
    prob = pulp.LpProblem("GHP_integer", pulp.LpMinimize)

    # Decision variables x_i_j binary
    x = pulp.LpVariable.dicts("x", (range(n_f), range(n_s)), lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: sum rf[i] * (slot_time - ETA_i) * x[i,j]
    prob += pulp.lpSum([rf[i] * (s_index[j] - ETA_minutes[i]) * x[i][j]
                        for (i, j) in feasible_pairs])

    # Constraints:
    # 1) Each flight assigned exactly 1 slot (over feasible j)
    for i in range(n_f):
        feasible_js = [j for (ii, j) in feasible_pairs if ii == i]
        prob += pulp.lpSum([x[i][j] for j in feasible_js]) == 1, f"Assign_flight_{i}"

    # 2) Each slot at most 1 flight
    for j in range(n_s):
        feasible_is = [i for (i, jj) in feasible_pairs if jj == j]
        if feasible_is:
            prob += pulp.lpSum([x[i][j] for i in feasible_is]) <= 1, f"Slot_capacity_{j}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)  # msg=True to see solver log
    result = prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    print("LP solve status:", status)

    if status not in ("Optimal", "Integer Feasible"):
        print("Warning: solver did not find optimal solution. Status:", status)
        # still try to extract assignments if any

    # Reset assigned fields for all flights
    for f in filtered_arrivals:
        f.assigned_slot_time = None
        f.assigned_delay = 0
        f.original_eta_minutes = time_to_minutes(f.arr_time)

    # For flights not needing slots, keep original
    for f in flights_not_needing:
        f.assigned_slot_time = time_to_minutes(f.arr_time)
        f.assigned_delay = 0

    # Read variables and assign slots
    assigned_count = 0
    used_slots = set()
    for i in range(n_f):
        for j in range(n_s):
            try:
                val = pulp.value(x[i][j])
            except Exception:
                val = None
            if val is not None and val > 0.5:
                slot_time = s_index[j]
                flight_obj = flights_needing_slots[i]
                flight_obj.assigned_slot_time = slot_time
                flight_obj.assigned_delay = slot_time - ETA_minutes[i]
                assigned_count += 1
                used_slots.add(j)
                break  # flight assigned, next flight

    print(f"Assigned {assigned_count}/{n_f} regulated flights to slots (objective: {objective})")

    # compute total statistics similar to print_delay_statistics
    return filtered_arrivals



if __name__ == "__main__":
    print("You're not executing the main program")