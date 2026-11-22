from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors

from services import llm_client

_SERVER_ERROR_BODY = {
    "error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}
}


def _fake_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


def _server_error() -> errors.ServerError:
    return errors.ServerError(503, _SERVER_ERROR_BODY)


@patch("services.llm_client.time.sleep")
@patch("services.llm_client._get_client")
def test_retries_once_on_transient_server_error_then_succeeds(mock_get_client, mock_sleep):
    fake_client = MagicMock()
    ok_json = (
        '{"reply": "ok", "urgency": "green", "possible_causes": [], '
        '"recommended_specialty": null, "suggested_tests": [], '
        '"follow_up_questions": [], "emergency_flag": false}'
    )
    fake_client.models.generate_content.side_effect = [_server_error(), _fake_response(ok_json)]
    mock_get_client.return_value = fake_client

    result = llm_client.get_llm_response("hello", [])

    assert result["reply"] == "ok"
    assert fake_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()


@patch("services.llm_client.time.sleep")
@patch("services.llm_client._get_client")
def test_retries_the_primary_model_before_moving_on(mock_get_client, mock_sleep, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "primary-model")
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _server_error()
    mock_get_client.return_value = fake_client

    with pytest.raises(errors.ServerError):
        llm_client.get_llm_response("hello", [])

    primary_calls = [
        c for c in fake_client.models.generate_content.call_args_list
        if c.kwargs["model"] == "primary-model"
    ]
    assert len(primary_calls) == llm_client._PRIMARY_ATTEMPTS


@patch("services.llm_client._get_client")
def test_succeeds_immediately_without_retry_when_no_error(mock_get_client):
    fake_client = MagicMock()
    ok_json = (
        '{"reply": "ok", "urgency": "green", "possible_causes": [], '
        '"recommended_specialty": null, "suggested_tests": [], '
        '"follow_up_questions": [], "emergency_flag": false}'
    )
    fake_client.models.generate_content.return_value = _fake_response(ok_json)
    mock_get_client.return_value = fake_client

    result = llm_client.get_llm_response("hello", [])

    assert result["reply"] == "ok"
    assert fake_client.models.generate_content.call_count == 1


@patch("services.llm_client.time.sleep")
@patch("services.llm_client._get_client")
def test_falls_back_to_another_model_when_the_primary_is_unavailable(
    mock_get_client, mock_sleep, monkeypatch
):
    # A 503 from one model says nothing about the others. Overload has taken
    # the service down three times; retrying a dead model alone doesn't help.
    monkeypatch.setenv("MODEL_NAME", "primary-model")
    fake_client = MagicMock()
    ok_json = (
        '{"reply": "ok", "urgency": "green", "possible_causes": [], '
        '"recommended_specialty": null, "suggested_tests": [], '
        '"follow_up_questions": [], "emergency_flag": false}'
    )
    # Primary fails both attempts, first fallback succeeds.
    fake_client.models.generate_content.side_effect = [
        _server_error(),
        _server_error(),
        _fake_response(ok_json),
    ]
    mock_get_client.return_value = fake_client

    result = llm_client.get_llm_response("hello", [])

    assert result["reply"] == "ok"
    models_tried = [
        call.kwargs["model"] for call in fake_client.models.generate_content.call_args_list
    ]
    assert models_tried[0] == "primary-model"
    assert models_tried[-1] != "primary-model", "should have moved to a different model"


@patch("services.llm_client.time.sleep")
@patch("services.llm_client._get_client")
def test_raises_only_after_every_model_is_exhausted(mock_get_client, mock_sleep):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _server_error()
    mock_get_client.return_value = fake_client

    with pytest.raises(errors.ServerError):
        llm_client.get_llm_response("hello", [])

    tried = {
        call.kwargs["model"] for call in fake_client.models.generate_content.call_args_list
    }
    assert len(tried) > 1, "should have tried more than one model before giving up"
