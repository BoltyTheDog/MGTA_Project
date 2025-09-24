from Functions import initialiseflights, plot_flight_count

max_capacity: int = 40  # LEBL max capacity
HStart: int = 12 # Regulation start hour
HEnd: int = 18 # Regulation end hour

flights = initialiseflights("LEBL_10AUG2025.csv")

plot_flight_count(flights, max_capacity, HStart, HEnd)
