from work import knots_to_kmh, estimate_flight_time_hours


def test_knots_to_kmh_normal():
    assert knots_to_kmh(3) == 5.556
    assert knots_to_kmh(100) == 185.2


def test_knots_to_kmh_zero_and_negative():
    assert knots_to_kmh(0) == 0.0
    try:
        knots_to_kmh(-50)
        assert False
    except ValueError as e:
        assert str(e) == "Speed must be non-negative."


def test_estimate_flight_time_hours_normal():
    assert estimate_flight_time_hours(2350, 470) == 5.0
    assert estimate_flight_time_hours(1000, 250) == 4.0


def test_estimate_flight_time_hours_invalid():
    try:
        estimate_flight_time_hours(0, 400)
        assert False
    except ValueError as e:
        assert str(e) == "Distance must be positive."

    try:
        estimate_flight_time_hours(400, 0)
        assert False
    except ValueError as e:
        assert str(e) == "Speed must be positive."
