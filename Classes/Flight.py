from datetime import datetime, timedelta
import emissions_fuel_model as e

class Flight:
    def __init__(self, callsign: str, airplane_model: str, departure_airport: str, arrival_airport: str,
                 crz_fl: int, crz_spd: float, departure_time: datetime, taxi_time: timedelta,
                 arrival_time: datetime, flight_time: timedelta, flight_dis: float, category: str,
                 seats: int, is_ecac: bool = True, is_exempt: bool = False, delay_type: str = "") -> None:
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
        self.is_ecac = is_ecac  # New: ECAC status
        self.is_exempt = is_exempt  # Exemption status
        self.delay_type = delay_type  # Delay type

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
                f"Is ECAC: {self.is_ecac}\n"
                f"Is Exempt: {self.is_exempt}\n"
                f"Delay Type: {self.delay_type}\n")

    def computeunrecdel(self, delay: int, hstart: int) -> float:

        unrecoverabledelay: float = 0
        hstarttime = datetime.strptime(str(hstart), "%H")

        ctd = self.dep_time + timedelta(minutes=int(delay))
        if self.dep_time > hstarttime:
            unrecoverabledelay = 0
        elif ctd < hstarttime:
            unrecoverabledelay = delay
        elif self.dep_time < hstarttime < ctd:
            delaydiff = hstarttime - self.dep_time
            unrecoverabledelay = delaydiff.seconds / 60

        return unrecoverabledelay


    def compute_air_del_emissions(self, delay: int, objective: str) -> float: #return kg CO2/min in air delay flights (exempt flights)

        seats = self.seats
        velocity = self.cruise_spd
        distance = 0
        if objective == "delay":
            distance = self.cruise_spd*delay
        if objective == "flight":
            distance = self.flight_distance
        if objective == "total":
            distance = self.cruise_spd*delay + self.flight_distance

        if seats < 50:
            seats = 50
        elif seats > 365:
            seats = 365
        if distance < 100:
            distance = 100
        elif distance > 12000:
            distance = 12000



        # Use force=True to bypass strict validation for edge cases
        co2_ask = e.compute_co2_ask(distance, seats, force=True)

        total_co2 = co2_ask * seats * velocity * (1/1000) * 60 *(1/1000)
        return total_co2

    def compute_ground_del_emissions(self) -> float: #return kg CO2/min in air delay flights (exempt flights)
        fuel_consum = 0
        match self.cat:
            case "A":
                fuel_consum = 0
            case "B":
                fuel_consum = (50/60)
            case "C":
                fuel_consum = (75/60)
            case "D":
                fuel_consum = (110/60)
            case "E":
                fuel_consum = (170/60)
            case "F":
                fuel_consum = (260/60)
            case _:
                raise ValueError("Invalid Category")
        return fuel_consum * 3.16

    def compute_costs(self, delay: int) -> float:
        ...


if __name__ == "__main__":
    print("You're not executing the main program")