from datetime import datetime


class Flight:
    def __init__(self, callsign: str, airplane_model: str, departure_airport: str, arrival_airport: str, registration: str, crz_fl: int, departure_time: datetime, taxi_time: datetime, arrival_time: datetime, flight_time: datetime, category: str, seats: int) -> None:
        self.callsign = callsign
        self.airplane_model = airplane_model
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
        self.registration = registration
        self.cruise = crz_fl
        self.dep_time = departure_time
        self.taxi_time = taxi_time
        self.arr_time = arrival_time
        self.flight_time = flight_time
        self.cat = category
        self.seats = seats

    def __str__(self):
        return (f"Flight info:\n"
                f"Callsign : {self.callsign}\n"
                f"Airplane: {self.airplane_model}\n"
                f"Dep airport: {self.departure_airport}\n"
                f"Arr airport: {self.arrival_airport}\n"
                f"Registration: {self.registration}\n"
                f"Cruise FL: {self.cruise}\n"
                f"Departure time: {self.dep_time}\n"
                f"Taxi time: {self.taxi_time}\n"
                f"Arrival time: {self.arr_time}\n"
                f"Flight time: {self.flight_time}\n"
                f"Category: {self.cat}\n"
                f"Seats No: {self.seats}\n")

if __name__ == "__main__":
    print("You're not executing the main program")


