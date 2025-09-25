from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
from Classes.Flight import Flight

def initialise_flights(filename: str) -> list[Flight] | None:
    flights = []
    with open(filename, 'r') as r:
        next(r)
        for line in r:
            line_array = line.split(";")
            if line_array[3] == "LEBL":
                dep_time = datetime.strptime(line_array[6], "%H:%M:%S")
                arr_time = datetime.strptime(line_array[8], "%H:%M:%S")
                taxi_time = datetime.strptime(line_array[7], "%M")
                flight_time = datetime.strptime(line_array[11], "%H:%M:%S")


                flights.append(Flight(line_array[0], line_array[1], line_array[2], line_array[3],
                                      line_array[4], int(line_array[5]), dep_time, taxi_time,
                                      arr_time, flight_time, line_array[12], int(line_array[13])))
    if len(flights) == 0:
        return None

    return flights

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

if __name__ == "__main__":
    print("You're not executing the main program")