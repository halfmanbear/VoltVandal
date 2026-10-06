from voltvandal.core import flightlog


def test_flight_log_detects_unfinished_previous_run(tmp_path):
    assert flightlog.start(tmp_path) is None
    flightlog.log("candidate_begin", label="x")
    again = flightlog.start(tmp_path)  # no session_end was written
    assert again and "candidate_begin" in again
    flightlog.log("session_end")
    assert flightlog.start(tmp_path) is None
