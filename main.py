from Functions import initialise_flights, plot_flight_count, plot_aggregated_demand, dime_cantidad_aerolinea_por_hora, computeSlots

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

Hstart_min = 648  # 10:48 in minutes
Hend_min = 780   # 13:00 in minutes
HNoReg_min = 660 # 11:00 in minutes
PAAR = reduced_capacity
AAR = max_capacity
slots = computeSlots(Hstart_min, Hend_min, HNoReg_min, PAAR, AAR)

cantidadVLG = dime_cantidad_aerolinea_por_hora(arrival_flights, "VLG",10, 11)
print(cantidadVLG)

print("Computed slots matrix:")
print(slots)

