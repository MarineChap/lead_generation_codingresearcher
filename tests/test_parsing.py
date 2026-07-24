"""Tests for robust crew-output JSON parsing."""

from lead_gen.parsing import parse_json_array


def test_bare_array():
    result = parse_json_array('[{"a": 1}]', context="t")
    assert not result.failed
    assert result.items == [{"a": 1}]


def test_fenced_json_block():
    raw = 'Here is the result:\n```json\n[{"x": 2}]\n```\nDone.'
    result = parse_json_array(raw, context="t")
    assert result.items == [{"x": 2}]


def test_array_embedded_in_prose():
    raw = 'The leads are [{"n": "lab"}] and that is all.'
    result = parse_json_array(raw, context="t")
    assert result.items == [{"n": "lab"}]


def test_single_key_dict_wrapping_list():
    result = parse_json_array('{"leads": [{"i": 1}]}', context="t")
    assert result.items == [{"i": 1}]


def test_empty_array_is_not_a_failure():
    result = parse_json_array("[]", context="t")
    assert not result.failed
    assert result.items == []


def test_garbage_is_a_failure_not_silent_empty():
    result = parse_json_array("the model refused to answer", context="t")
    assert result.failed
    assert result.error is not None
    assert result.items == []


def test_retry_callback_recovers():
    calls = {"n": 0}

    def retry():
        calls["n"] += 1
        return '[{"recovered": true}]'

    result = parse_json_array("broken", context="t", retry_cb=retry)
    assert calls["n"] == 1
    assert result.items == [{"recovered": True}]


def test_retry_also_failing_reports_error():
    result = parse_json_array("broken", context="t", retry_cb=lambda: "still broken")
    assert result.failed
