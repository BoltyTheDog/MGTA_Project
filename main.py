from Functions import initialise_flights, plot_flight_count, plot_aggregated_demand, amount_flights_by_hour, compute_slots, flight_distance, flight_by_callsign

max_capacity: int = 40  # LEBL max capacity
reduced_capacity: int = 20  # LEBL reduced capacity
HStart: int = 11 # Regulation start hour
HEnd: int = 18 # Regulation end hour

arrival_flights = initialise_flights("Data/LEBL_10AUG2025_ECAC.csv")

print(arrival_flights[0])
if not arrival_flights:
    print("There are no flights to display")
    exit(1)

plot_flight_count(arrival_flights, max_capacity, HStart, HEnd)

plot_aggregated_demand(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)

Hstart_min: int = 648  # 10:48 in minutes
Hend_min: int = 780   # 13:00 in minutes
HNoReg_min: int = 660 # 11:00 in minutes
PAAR: int = reduced_capacity
AAR: int = max_capacity
slots = compute_slots(Hstart_min, Hend_min, HNoReg_min, PAAR, AAR)

amountVLG = amount_flights_by_hour(arrival_flights, "VLG",10, 11)
print(amountVLG)

print("Computed slots matrix:")
print(slots)

testflight = flight_by_callsign(arrival_flights, "RYR404A")

distance = flight_distance(testflight)
print(distance)
print(type(distance))