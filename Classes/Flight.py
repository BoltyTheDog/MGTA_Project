class Flight:
    def __init__(self, callsign: str, airplane_model: str, departure_airport: str, arrival_airport: str, registration: str, crz_fl: int, departure_time: str) -> None:
        self.callsign = callsign
        self.airplane_model = airplane_model
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
