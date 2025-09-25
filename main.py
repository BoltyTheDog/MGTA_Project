from Functions import initialise_flights, plot_flight_count

max_capacity: int = 40  # LEBL max capacity
HStart: int = 12 # Regulation start hour
HEnd: int = 18 # Regulation end hour

arrival_flights = initialise_flights("LEBL_10AUG2025.csv")

print(arrival_flights[0])
if not arrival_flights:
    print("There are no flights to display")
    exit(1)

plot_flight_count(arrival_flights, max_capacity, HStart, HEnd)
