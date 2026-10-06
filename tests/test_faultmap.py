from datetime import datetime, timezone

from voltvandal.core import faultmap
from voltvandal.hardware.events import FaultWatch, parse_wevtutil_xml


def test_in_flight_point_is_blocked_after_crash(tmp_path):
    faultmap.mark_in_flight(tmp_path, "a", 881250, 2130000)
    assert faultmap.recover(tmp_path)["freq_khz"] == 2130000
    assert faultmap.recover(tmp_path) is None
    assert faultmap.is_blocked(tmp_path, 881250, 2130000)
    assert faultmap.is_blocked(tmp_path, 850000, 2200000)
    assert not faultmap.is_blocked(tmp_path, 881250, 2085000)
    assert not faultmap.is_blocked(tmp_path, 900000, 2130000)


def test_clean_run_leaves_nothing_blocked(tmp_path):
    faultmap.mark_in_flight(tmp_path, "a", 881250, 2085000)
    faultmap.clear_in_flight(tmp_path)
    assert faultmap.recover(tmp_path) is None
    assert not faultmap.is_blocked(tmp_path, 881250, 2085000)


def test_record_fault_keeps_lowest_frequency(tmp_path):
    faultmap.record_fault(tmp_path, 881250, 2130000)
    faultmap.record_fault(tmp_path, 881250, 2100000)
    faultmap.record_fault(tmp_path, 881250, 2200000)
    assert faultmap.failed_floors(tmp_path) == {881250: 2100000}


def test_parse_wevtutil_xml_filters_hardware_events():
    xml = ("<Event xmlns='x'><System><Provider Name='nvlddmkm'/><EventID Qualifiers='0'>14</EventID></System></Event>"
           "<Event xmlns='x'><System><Provider Name='Other' Guid='{1}'/><EventID>1</EventID></System></Event>")
    assert parse_wevtutil_xml(xml) == ["nvlddmkm#14"]
    assert parse_wevtutil_xml("") == []


def test_fault_watch_fires_callback_once():
    seen = []
    calls = iter([None, [], ["nvlddmkm#14"], ["nvlddmkm#14"]])
    w = FaultWatch(seen.append, query=lambda start: next(calls), interval_s=0.01).start()
    assert w.tripped.wait(2.0)
    w.stop()
    assert seen == [["nvlddmkm#14"]] and w.faults == ["nvlddmkm#14"]
