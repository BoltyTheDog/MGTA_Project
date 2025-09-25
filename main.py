from Functions import initialise_flights, plot_flight_count, plot_aggregated_demand, dime_cantidad_aerolinea_por_hora

max_capacity: int = 40  # LEBL max capacity
reduced_capacity: int = 20  # LEBL reduced capacity
HStart: int = 11 # Regulation start hour
HEnd: int = 18 # Regulation end hour

arrival_flights = initialise_flights("LEBL_10AUG2025.csv")

print(arrival_flights[0])
if not arrival_flights:
    print("There are no flights to display")
    exit(1)

plot_flight_count(arrival_flights, max_capacity, HStart, HEnd)
plot_aggregated_demand(arrival_flights, HStart, HEnd, max_capacity, reduced_capacity)


cantidadVLG = dime_cantidad_aerolinea_por_hora(arrival_flights, "VLG",10, 11)
print(cantidadVLG)
