import Functions as f
max_capacity: int = 40  # LEBL max capacity
reduced_capacity: int = 20  # LEBL reduced capacity
HStart: int = 11 # Regulation start hour
HEnd: int = 18 # Regulation end hour
HFile: int = 9 # Filing Hour
distThreshold: int = 3500 # Radius in Km

arrival_flights = f.initialise_flights("Data/LEBL_10AUG2025_ECAC.csv")

print(arrival_flights[0])
if not arrival_flights:
    print("There are no flights to display")
    exit(1)

f.plot_flight_count(arrival_flights, max_capacity, HStart, HEnd)

HNoReg: float = f.plot_aggregated_demand(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)

filtered_flights = f.filter_arrival_flights(arrival_flights, distThreshold, HStart, HNoReg, HFile)

initial_flights = filtered_flights.copy()


Hstart_min: int = HStart*60  # 11:00 in minutes
Hend_min: int = HEnd * 60   # 18:00 in minutes

HNoReg_min = HNoReg * 60  # Convert to minutes
extended_HNoReg = HNoReg_min + 30
slots = f.compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)

slotted_arrivals = f.assignSlotsGDP(filtered_flights, slots)

# Enforce capacity after initial assignment (may move flights to keep hourly capacity)
f.enforce_capacity(slots, slotted_arrivals, HStart, HEnd, reduced_capacity, max_capacity)
f.plot_slotted_arrivals(slotted_arrivals, max_capacity, HStart, HEnd)

f.print_delay_statistics(slotted_arrivals)



# Print basic slot assignments for all arrivals
print("\n" + "="*80)
print("BASIC SLOT ASSIGNMENTS - ALL ARRIVALS")
print("="*80)
print(f"{'#':<4} {'Callsign':<8} {'Original ETA':<12} {'Assigned Slot':<13} {'Delay (min)':<11} {'Type':<6} {'Exempt':<6}")
print("-" * 80)

# Sort flights by assigned slot time for display; place unassigned flights at the end
sorted_flights = sorted(slotted_arrivals, key=lambda fl: (fl.assigned_slot_time is None, fl.assigned_slot_time if fl.assigned_slot_time is not None else 10**9))
unrecoverabledelay: float = 0
otpcounter: int = 0

air_del_emission_count = 0
ground_del_emission_count = 0

rf_GDP = []

for i, flight in enumerate(sorted_flights, 1):
    original_eta = flight.arr_time.strftime('%H:%M:%S')
    delay = 0
    if hasattr(flight, 'assigned_slot_time') and flight.assigned_slot_time is not None:
        assigned_slot = f"{flight.assigned_slot_time//60:02d}:{flight.assigned_slot_time%60:02d}:00"
        delay = getattr(flight, 'assigned_delay', 0)
    else:
        assigned_slot = original_eta

    delay_type = getattr(flight, 'delay_type', 'None')
    is_exempt = getattr(flight, 'is_exempt', False)

    unrecoverabledelay += flight.computeunrecdel(delay, HStart)

    if delay < 15:
        otpcounter += 1
    
    print(f"{i:<4} {flight.callsign:<8} {original_eta:<12} {assigned_slot:<13} {delay:<11} {delay_type:<6} {is_exempt}")

    if flight.delay_type == "Air":
        air_emission = flight.compute_air_del_emissions(delay, "delay") * delay
        air_del_emission_count += air_emission
        rf_GDP.append(air_emission)
    elif flight.delay_type == "Ground":
        ground_emission = flight.compute_ground_del_emissions(delay)
        ground_del_emission_count += flight.compute_ground_del_emissions(delay)
        rf_GDP.append(ground_emission)

print("="*80)
print(f"Unrecoverable delay = {unrecoverabledelay} mins")
print("="*80)
print(f"# of flights with 15+ minutes of delay: {otpcounter}")
print("="*80)
print("Total of emissions/min from air delay:", air_del_emission_count, "kg of CO2")
print("Total of emissions/min from ground delay:", ground_del_emission_count, "kg of CO2")
print("="*80)

# Cancel top-10 VLG flights and re-run slotting prioritizing VLG; print comparison
print("\nApplying cancellations (top-10 VLG delays) and re-slotting with VLG priority...")
slotted_arrivals = f.cancel_and_reslot(slotted_arrivals, slots, company='VLG', n_cancel=10)



# Plot and print statistics after reslotting
# Enforce capacity again after the reslot operation to ensure no hour exceeds limits
f.enforce_capacity(slots, slotted_arrivals, HStart, HEnd, reduced_capacity, max_capacity)
f.plot_slotted_arrivals(slotted_arrivals, max_capacity, HStart, HEnd)
f.print_delay_statistics(slotted_arrivals)

# Print basic slot assignments for all arrivals
print("\n" + "=" * 80)
print("BASIC SLOT ASSIGNMENTS - ALL ARRIVALS")
print("=" * 80)
print(
    f"{'#':<4} {'Callsign':<8} {'Original ETA':<12} {'Assigned Slot':<13} {'Delay (min)':<11} {'Type':<6} {'Exempt':<6}")
print("-" * 80)

# Sort flights by assigned slot time for display; place unassigned flights at the end
sorted_flights = sorted(slotted_arrivals, key=lambda fl: (fl.assigned_slot_time is None,
                                                          fl.assigned_slot_time if fl.assigned_slot_time is not None else 10 ** 9))
unrecoverabledelay: float = 0
otpcounter: int = 0

air_del_emission_count = 0
ground_del_emission_count = 0

rf_GDP = []

for i, flight in enumerate(sorted_flights, 1):
    original_eta = flight.arr_time.strftime('%H:%M:%S')
    delay = 0
    if hasattr(flight, 'assigned_slot_time') and flight.assigned_slot_time is not None:
        assigned_slot = f"{flight.assigned_slot_time // 60:02d}:{flight.assigned_slot_time % 60:02d}:00"
        delay = getattr(flight, 'assigned_delay', 0)
    else:
        assigned_slot = original_eta

    delay_type = getattr(flight, 'delay_type', 'None')
    is_exempt = getattr(flight, 'is_exempt', False)

    unrecoverabledelay += flight.computeunrecdel(delay, HStart)

    if delay < 15:
        otpcounter += 1

    print(f"{i:<4} {flight.callsign:<8} {original_eta:<12} {assigned_slot:<13} {delay:<11} {delay_type:<6} {is_exempt}")

    if flight.delay_type == "Air":
        air_emission = flight.compute_air_del_emissions(delay, "delay") * delay
        air_del_emission_count += air_emission
        rf_GDP.append(air_emission)
    elif flight.delay_type == "Ground":
        ground_emission = flight.compute_ground_del_emissions(delay)
        ground_del_emission_count += flight.compute_ground_del_emissions(delay)
        rf_GDP.append(ground_emission)

print("=" * 80)
print(f"Unrecoverable delay = {unrecoverabledelay} mins")
print("=" * 80)
print(f"# of flights with 15+ minutes of delay: {otpcounter}")
print("=" * 80)
print("Total of emissions/min from air delay:", air_del_emission_count, "kg of CO2")
print("Total of emissions/min from ground delay:", ground_del_emission_count, "kg of CO2")
print("=" * 80)

print("COMPUTING GHP ANALYSIS...")
# (unitary cost => rf = 1) -> total cost = total delay
f.compute_GHP(filtered_flights, slots, HStart, objective='costs')

print("="*80)
print("="*80)
print("="*80)
print("REMOVING FLIGHTS THAT HAVE A TRAIN ALTERNATIVE")
print("="*80)
new_flights, _, _ = f.compute_Flight_Emissions_D2DTime(initial_flights)

print("FLights eliminated = ", str(len(filtered_flights) - len(new_flights)))

print("="*80)
print("REEXECUTING GDP AND GHP")

HNoReg = f.plot_aggregated_demand(new_flights, HStart, HEnd, max_capacity, reduced_capacity)

f.plot_flight_count(new_flights, max_capacity, HStart, HEnd)


slots = f.compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)



slotted_arrivals = f.assignSlotsGDP(new_flights, slots)

# Enforce capacity after initial assignment (may move flights to keep hourly capacity)
f.enforce_capacity(slots, slotted_arrivals, HStart, HEnd, reduced_capacity, max_capacity)
f.plot_slotted_arrivals(slotted_arrivals, max_capacity, HStart, HEnd)

f.print_delay_statistics(slotted_arrivals)


# Print basic slot assignments for all arrivals
print("\n" + "=" * 80)
print("BASIC SLOT ASSIGNMENTS - ALL ARRIVALS")
print("=" * 80)
print(
    f"{'#':<4} {'Callsign':<8} {'Original ETA':<12} {'Assigned Slot':<13} {'Delay (min)':<11} {'Type':<6} {'Exempt':<6}")
print("-" * 80)

# Sort flights by assigned slot time for display; place unassigned flights at the end
sorted_flights = sorted(slotted_arrivals, key=lambda fl: (fl.assigned_slot_time is None,
                                                          fl.assigned_slot_time if fl.assigned_slot_time is not None else 10 ** 9))
unrecoverabledelay: float = 0
otpcounter: int = 0

air_del_emission_count = 0
ground_del_emission_count = 0

rf_GDP = []

for i, flight in enumerate(sorted_flights, 1):
    original_eta = flight.arr_time.strftime('%H:%M:%S')
    delay = 0
    if hasattr(flight, 'assigned_slot_time') and flight.assigned_slot_time is not None:
        assigned_slot = f"{flight.assigned_slot_time // 60:02d}:{flight.assigned_slot_time % 60:02d}:00"
        delay = getattr(flight, 'assigned_delay', 0)
    else:
        assigned_slot = original_eta

    delay_type = getattr(flight, 'delay_type', 'None')
    is_exempt = getattr(flight, 'is_exempt', False)

    unrecoverabledelay += flight.computeunrecdel(delay, HStart)

    if delay < 15:
        otpcounter += 1

    print(f"{i:<4} {flight.callsign:<8} {original_eta:<12} {assigned_slot:<13} {delay:<11} {delay_type:<6} {is_exempt}")

    if flight.delay_type == "Air":
        air_emission = flight.compute_air_del_emissions(delay, "delay") * delay
        air_del_emission_count += air_emission
        rf_GDP.append(air_emission)
    elif flight.delay_type == "Ground":
        ground_emission = flight.compute_ground_del_emissions(delay)
        ground_del_emission_count += flight.compute_ground_del_emissions(delay)
        rf_GDP.append(ground_emission)

print("=" * 80)
print(f"Unrecoverable delay = {unrecoverabledelay} mins")
print("=" * 80)
print(f"# of flights with 15+ minutes of delay: {otpcounter}")
print("=" * 80)
print("Total of emissions/min from air delay:", air_del_emission_count, "kg of CO2")
print("Total of emissions/min from ground delay:", ground_del_emission_count, "kg of CO2")
print("=" * 80)

print("COMPUTING GHP ANALYSIS...")
# (unitary cost => rf = 1) -> total cost = total delay
f.compute_GHP(new_flights, slots, HStart, objective='costs')




# Plot HFile analysis graph
print("\n" + "="*80)
print("GENERATING HFILE ANALYSIS GRAPH")
print("="*80)
f.plot_hfile_analysis(arrival_flights, distThreshold, HStart, HEnd, reduced_capacity, max_capacity)


# Plot Distance Threshold analysis graph
print("\n" + "="*80)
print("GENERATING DISTANCE THRESHOLD ANALYSIS GRAPH")
print("="*80)
f.plot_distance_threshold_analysis(arrival_flights, HFile, HStart, HEnd, reduced_capacity, max_capacity)


# Plot 3D analysis with HFile vs Distance Threshold
print("\n" + "="*80)
print("GENERATING 3D ANALYSIS: HFILE vs DISTANCE THRESHOLD")
print("="*80)
f.plot_3d_analysis(arrival_flights, HStart, HEnd, reduced_capacity, max_capacity)
