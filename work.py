def knots_to_kmh(knots: float) -> float:
    """
    Convert speed from knots to km/h.
    1 knot = 1.852 km/h
    """
    if knots < 0:
        raise ValueError("Speed must be non-negative.")
    return round(knots * 1.852, 3)


def estimate_flight_time_hours(distance_nm: float,
                               speed_knots: float) -> float:
    """
    Estimate cruise flight time in hours.
    distance_nm: distance in nautical miles
    speed_knots: true airspeed in knots (nm per hour)
    """
    if distance_nm <= 0:
        raise ValueError("Distance must be positive.")
    if speed_knots <= 0:
        raise ValueError("Speed must be positive.")
    hours = distance_nm / speed_knots
    return round(hours, 3)
