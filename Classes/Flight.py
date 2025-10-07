from datetime import datetime, timedelta


class Flight:
    def __init__(self, callsign: str, airplane_model: str, departure_airport: str, arrival_airport: str, crz_fl: int, crz_spd: float, departure_time: datetime, taxi_time: timedelta, arrival_time: datetime, flight_time: timedelta, flight_dis: float, category: str, seats: int, exempt: str, del_type: str) -> None:
        self.callsign = callsign
        self.airplane_model = airplane_model
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
        self.cruise_fl = crz_fl
        self.cruise_spd = crz_spd
        self.dep_time = departure_time
        self.taxi_time = taxi_time
        self.arr_time = arrival_time
        self.flight_time = flight_time
        self.flight_distance = flight_dis
        self.cat = category
        self.seats = seats
        self.exempt = exempt
        self.delay_type = del_type

    def __str__(self):
        return (f"Flight info:\n"
                f"Callsign : {self.callsign}\n"
                f"Airplane: {self.airplane_model}\n"
                f"Dep airport: {self.departure_airport}\n"
                f"Arr airport: {self.arrival_airport}\n"
                f"Cruise FL: {self.cruise_fl}\n"
                f"Cruise speed: {self.cruise_spd} m/s\n"
                f"Departure time: {self.dep_time}\n"
                f"Taxi time: {self.taxi_time}\n"
                f"Arrival time: {self.arr_time}\n"
                f"Flight time: {self.flight_time}\n"
                f"Flight distance: {self.flight_distance} km\n"
                f"Category: {self.cat}\n"
                f"Seats No: {self.seats}\n"
                f"Exempt: {self.exempt}\n"
                f"Delay type: {self.delay_type}\n")



if __name__ == "__main__":
    print("You're not executing the main program")


