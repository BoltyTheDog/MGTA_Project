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

Hstart_min: int = HStart*60  # 11:00 in minutes
Hend_min: int = HEnd * 60   # 18:00 in minutes

HNoReg_min = HNoReg * 60  # Convert to minutes
extended_HNoReg = HNoReg_min + 30
slots = f.compute_slots(Hstart_min, Hend_min, extended_HNoReg, reduced_capacity, max_capacity)

# amountVLG = f.amount_flights_by_hour(arrival_flights, "VLG",10, 11)
# print(amountVLG)

slotted_arrivals = f.assignSlotsGDP(filtered_flights, slots)

f.plot_slotted_arrivals(slotted_arrivals, max_capacity, HStart, HEnd)

f.print_delay_statistics(slotted_arrivals)

# Print basic slot assignments for all arrivals
print("\n" + "="*80)
print("BASIC SLOT ASSIGNMENTS - ALL ARRIVALS")
print("="*80)
print(f"{'#':<4} {'Callsign':<8} {'Original ETA':<12} {'Assigned Slot':<13} {'Delay (min)':<11} {'Type':<6} {'Exempt':<6}")
print("-" * 80)

# Sort flights by original ETA for display
sorted_flights = sorted(slotted_arrivals, key=lambda f: f.arr_time)

for i, flight in enumerate(sorted_flights, 1):
    original_eta = flight.arr_time.strftime('%H:%M:%S')
    
    if hasattr(flight, 'assigned_slot_time') and flight.assigned_slot_time is not None:
        assigned_slot = f"{flight.assigned_slot_time//60:02d}:{flight.assigned_slot_time%60:02d}:00"
        delay = getattr(flight, 'assigned_delay', 0)
    else:
        assigned_slot = original_eta
        delay = 0
    
    delay_type = getattr(flight, 'delay_type', 'None')
    is_exempt = getattr(flight, 'is_exempt', False)
    
    print(f"{i:<4} {flight.callsign:<8} {original_eta:<12} {assigned_slot:<13} {delay:<11} {delay_type:<6} {is_exempt}")

print("="*80)
