from datetime import datetime, timedelta
from typing import Literal

import emissions_fuel_model as e
import pandas as pd

# DEFINITION OF THE CLASS FLIGHT CONTAINING ALL THE DATA INFORMATION FROM THE EXCELL OF THE STARTING POINT OF THE PROJECT.
class Flight:
    def __init__(self, callsign: str, airplane_model: str, departure_airport: str, arrival_airport: str,
                 crz_fl: int, crz_spd: float, departure_time: datetime, taxi_time: timedelta,
                 arrival_time: datetime, flight_time: timedelta, flight_dis: float, category: str,
                 seats: int, registration: str, is_ecac: bool = True, is_exempt: bool = False, delay_type: str = ""
                 ) -> None:
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
        self.registration = registration

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


    #FUNCTION THAT COMPUTES THE UNRECOVERABLE DELAY OF AN AIRCRAFT RECIEVING THE DELAY OF THE FLIGHT AND THE TIME THE REGULATION STARTS AS INPUTS.
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


    #FUNCTION TO COMPUTE THE AIR DELAY EMISSIONS USING THE GIVEN FUNCTION BY THE TEACHER.
    def compute_air_del_emissions(self, delay: int, objective = Literal["delay", "flight", "total"]) -> float: #return kg CO2/min in air delay flights (exempt flights)

        seats = self.seats
        velocity = self.cruise_spd
        distance = 0
        if objective == "delay": #WE WANT TO COMPUTE THE CO2 OF ONLY THE DELAY DISTANCE
            distance = self.cruise_spd*delay
        if objective == "flight": #WE WANT TO COMPUTE THE CO2 OF ONLY THE FLIGHT DISTANCE (WITHOUD DELAY)
            distance = self.flight_distance
        if objective == "total": #WE WANT TO COMPUTE THE CO2 OF THE WHOLE FLIGHT (FLIGHT DISTANCE + DELAY)
            distance = self.cruise_spd*delay + self.flight_distance

        #WE SET SOME FIX VALUES TO THOSE AIRCRAFT THAT DO NOT FOLLOW THE RESTRICTIONS OF THE FUNCTION GIVEN BY THE TEACHER
        if seats < 50: #minimum available seats for the function are 50.
            seats = 50
        elif seats > 365: #maximum available seats for the function are 365
            seats = 365
        if distance < 100: #minimum distance should be 100km
            distance = 100
        elif distance > 12000: #maximum distance should be 12000km
            distance = 12000


        # Use force=True to bypass strict validation for edge cases
        co2_ask = e.compute_co2_ask(distance, seats, force=True)

        #we first have the value of the function in gCO2/ASK so we transform it into kgC02/min
        total_co2 = co2_ask * seats * velocity * (1/1000) * 60 *(1/1000)
        return total_co2


    #COMPUTATION OF GROUND DELAY EMISSIONS PER CATEGORY OF AIRCRAFT
    def compute_ground_del_emissions(self) -> float:  # return kg CO2/min in air delay flights (exempt flights)
        fuel_consum = 0
        match self.cat:  # depending on the category of the aircraft, when at APU consumes X amount of fuel by hour so dividing by 60 we get kg fuel/min
            case "A":
                fuel_consum = (260 / 60)
            case "B":
                fuel_consum = (170 / 60)
            case "C":
                fuel_consum = (110 / 60)
            case "D":
                fuel_consum = (75 / 60)
            case "E":
                fuel_consum = (50 / 60)
            case "F":
                fuel_consum = 0
            case _:
                raise ValueError("Invalid Category")
        return fuel_consum * 3.16

    def cost_number(self, costs: pd.DataFrame, delay: int) -> float:
        if costs.empty:  # panda dictionary to take the values from table
            return 0.0
        m = 1
        n = 1
        delay_thresholds = costs.columns[1:].astype(int).values  # obtain a vector with the threshold values of delay from the tables
        cost_thresholds = costs[costs['Delay'] == self.cat].iloc[0, 1:].astype(int).tolist()  # obtain the values of the row associated to cat A,B,C...

        # computation of the cost linearly interpolating as a Straight Line Equation y=mx+n
        if delay < 5:
            m = cost_thresholds[0] / delay_thresholds[0]
            n = cost_thresholds[0] - m * delay_thresholds[0]
        elif delay > 300:
            m = (cost_thresholds[-1] - cost_thresholds[-2]) / (delay_thresholds[-1] - delay_thresholds[-2])
            n = cost_thresholds[-1] - m * delay_thresholds[-1]

        else:
            for i in range(1, len(delay_thresholds) - 1):
                if delay_thresholds[i] < delay < delay_thresholds[i + 1]:
                    m = (cost_thresholds[i + 1] - cost_thresholds[i]) / (delay_thresholds[i + 1] - delay_thresholds[i])
                    n = cost_thresholds[i + 1] - m * delay_thresholds[i + 1]

        return m * delay + n  # return of the cost (y value of our straight line equation)

    def compute_costs(self, delay: int, air_costs: pd.DataFrame, ground_no_reac_costs: pd.DataFrame, ground_costs: pd.DataFrame, flights: pd.DataFrame) -> float:
        costs = None
        if self.delay_type == "Air":  # look for if delay type is Air
            costs = air_costs  # if air delay use air delay table
        elif self.delay_type == "Ground":  # look for if delay type is Ground
            # if ground delay look for if reactionary or not reactionary delay
            # to know if reactionary delay search if there is turn around flight
            matching_reg = flights[flights['RM'] == self.registration]  # Is there any other flight with the same RM (same plane, thus)?
            filtered_reg = matching_reg[matching_reg['ARCID'] != self.callsign]
            if filtered_reg.empty:  # If not a turn around flight, not reactionary delay
                costs = ground_no_reac_costs  # if not reactionary delay go to table of not RD at gate
            else:
                # if there do is any other flight operated by same aircraft
                # seek for the first flight departing earlier
                flight_etd = filtered_reg["ETD"].to_string(index=False)
                flight_etd = flight_etd.split("\n")
                etd_time = None
                for flight in flight_etd:
                    time_obj = datetime.strptime(flight.strip(), "%H:%M:%S").time()
                    # look for if there is a departing flight after the landing for the same aircraft
                    if time_obj > self.arr_time.time():
                        etd_time = time_obj
                        break
                # if there is not a turn around flight, costs with no reactionary delay
                if etd_time is None:
                    costs = ground_no_reac_costs
                # if there is a turn around flight, compute the delay compared to the turn around time for and aircraft
                else:
                    def time_to_minutes(t):
                        return t.hour * 60 + t.minute + t.second / 60

                    dep_minutes = time_to_minutes(self.dep_time.time())
                    etd_minutes = time_to_minutes(etd_time)

                    match self.cat:  # SWITCH CASE for aircraft categories for the TURN AROUND TIME value

                        # higher categories => 134 min of turn around time
                        case "A" | "B":
                            if dep_minutes + delay > etd_minutes - 134:
                                costs = ground_costs  # if CTA grater than ETD of the next flight, WE DO HAVE REACTIONARY DELAY
                            else:
                                costs = ground_no_reac_costs

                        # lower categories => 59 min of turn around time
                        case "C" | "D" | "E" | "F":
                            if dep_minutes + delay > etd_minutes - 59:
                                costs = ground_costs # if CTA grater than ETD of the next flight, WE DO HAVE REACTIONARY DELAY
                            else:
                                costs = ground_no_reac_costs
                        case _:
                            print(self.cat)
                            raise ValueError("Invalid Category")

        return self.cost_number(costs, delay)


if __name__ == "__main__":
    print("You're not executing the main program")