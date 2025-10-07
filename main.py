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

Hstart_min: int = 648  # 10:48 in minutes
Hend_min: int = 780   # 13:00 in minutes
PAAR: int = reduced_capacity
AAR: int = max_capacity
slots = f.compute_slots(Hstart_min, Hend_min, HNoReg * 60, PAAR, AAR)

amountVLG = f.amount_flights_by_hour(arrival_flights, "VLG",10, 11)
print(amountVLG)

print("Computed slots matrix:")
print(slots)

# Print arrival flights nicely
print("=" * 80)
print("ARRIVAL FLIGHTS VECTOR")
print("=" * 80)
print(f"Total flights: {len(arrival_flights)}")
print("-" * 80)

for i, flight in enumerate(arrival_flights, 1):
    print(f"Flight {i}:")
    print(f"  Callsign: {flight.callsign}")
    print(f"  Aircraft: {flight.airplane_model}")
    print(f"  From: {flight.departure_airport} → To: {flight.arrival_airport}")
    print(f"  Arrival Time: {flight.arr_time.strftime('%H:%M:%S')}")
    print(f"  Departure Time: {flight.dep_time.strftime('%H:%M:%S')}")
    print(f"  Flight Distance: {flight.flight_distance} km")
    print(f"  Category: {flight.cat}")
    print(f"  Seats: {flight.seats}")
    print(f"  ECAC?: {flight.is_ecac}")
    if hasattr(flight, 'is_exempt'):
        print(f"  Exempt: {flight.is_exempt}")
    if hasattr(flight, 'delay_type'):
        print(f"  Delay Type: {flight.delay_type}")
    print("-" * 40)

print("=" * 80)



