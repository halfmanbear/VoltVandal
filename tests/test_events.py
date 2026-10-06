from voltvandal.hardware.events import is_hardware_error, parse_events


def test_filter_matches_whea_nvidia_and_tdr_only():
    assert is_hardware_error("Microsoft-Windows-WHEA-Logger", 17)
    assert is_hardware_error("nvlddmkm", 14)
    assert is_hardware_error("Display", 4101)
    assert not is_hardware_error("Display", 4102)
    assert not is_hardware_error("Service Control Manager", 7036)


def test_parse_handles_empty_single_and_list():
    assert parse_events("") == []
    assert parse_events('{"Id":17,"ProviderName":"Microsoft-Windows-WHEA-Logger"}') == [
        "Microsoft-Windows-WHEA-Logger#17"]
    raw = '[{"Id":14,"ProviderName":"nvlddmkm"},{"Id":1,"ProviderName":"Other"}]'
    assert parse_events(raw) == ["nvlddmkm#14"]
