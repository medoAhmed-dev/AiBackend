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
def test_gives_up_after_max_attempts(mock_get_client, mock_sleep):
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = _server_error()
    mock_get_client.return_value = fake_client

    with pytest.raises(errors.ServerError):
        llm_client.get_llm_response("hello", [])

    assert fake_client.models.generate_content.call_count == llm_client._MAX_ATTEMPTS


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
