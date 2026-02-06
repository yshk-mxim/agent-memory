"""Unit tests for demo/app.py Streamlit application logic.

Tests pure functions and session state initialization without requiring
a running Streamlit server or the actual streamlit package.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class _SessionState(dict):
    """Mimics Streamlit's SessionState: dict + attribute access."""

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


@pytest.fixture(autouse=True)
def _mock_streamlit():
    """Mock streamlit module so demo/app.py can be imported."""
    mock_st = MagicMock(spec=ModuleType)
    mock_st.session_state = _SessionState()

    # Mock st.fragment decorator — it should be a passthrough
    def fragment_decorator(*args, **kwargs):
        # Handle both @st.fragment and @st.fragment(run_every="0.5s")
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Direct decoration: @st.fragment
            return args[0]
        else:
            # Parametrized: @st.fragment(run_every=...)
            def wrapper(func):
                return func
            return wrapper

    mock_st.fragment = fragment_decorator

    sys.modules["streamlit"] = mock_st
    yield mock_st
    sys.modules.pop("streamlit", None)
    # Remove cached demo.app module so next test gets fresh import
    sys.modules.pop("demo.app", None)
    sys.modules.pop("demo", None)


def _import_app():
    """Import demo.app after mocking streamlit."""
    # Force re-import
    sys.modules.pop("demo.app", None)
    sys.modules.pop("demo", None)
    from demo import app
    return app


class TestGetCacheState:
    """Cache state transitions: COLD -> WARM -> HOT based on turn count."""

    def test_turn_0_is_cold(self) -> None:
        app = _import_app()
        label, _color = app.get_cache_state(0)
        assert label == "COLD"

    def test_turn_1_is_warm(self) -> None:
        app = _import_app()
        label, _color = app.get_cache_state(1)
        assert label == "WARM"

    def test_turn_2_is_warm(self) -> None:
        app = _import_app()
        label, _color = app.get_cache_state(2)
        assert label == "WARM"

    def test_turn_3_is_hot(self) -> None:
        app = _import_app()
        label, _color = app.get_cache_state(3)
        assert label == "HOT"

    def test_turn_100_is_hot(self) -> None:
        app = _import_app()
        label, _color = app.get_cache_state(100)
        assert label == "HOT"

    def test_each_state_has_distinct_color(self) -> None:
        app = _import_app()
        _, cold_color = app.get_cache_state(0)
        _, warm_color = app.get_cache_state(1)
        _, hot_color = app.get_cache_state(3)

        colors = {cold_color, warm_color, hot_color}
        assert len(colors) == 3, "Each cache state must have a unique color"


class TestInitSessionState:
    """Session state initialization produces agents with unique IDs."""

    def test_creates_4_agents(self, _mock_streamlit) -> None:
        app = _import_app()
        session = _mock_streamlit.session_state

        # Not yet initialized
        assert "initialized" not in session

        app.init_session_state()

        assert session.get("initialized") is True

        for i in range(4):
            prefix = f"agent_{i}"
            assert f"{prefix}_sid" in session
            assert f"{prefix}_messages" in session
            assert f"{prefix}_turn" in session
            assert f"{prefix}_metrics" in session

    def test_agent_ids_are_unique(self, _mock_streamlit) -> None:
        app = _import_app()
        session = _mock_streamlit.session_state

        app.init_session_state()

        sids = [session[f"agent_{i}_sid"] for i in range(4)]
        assert len(set(sids)) == 4, "All 4 agent session IDs must be unique"

    def test_agent_ids_contain_names(self, _mock_streamlit) -> None:
        app = _import_app()
        session = _mock_streamlit.session_state

        app.init_session_state()

        expected_names = ["alpha", "beta", "gamma", "delta"]
        for i, name in enumerate(expected_names):
            sid = session[f"agent_{i}_sid"]
            assert name in sid, f"Agent {i} SID '{sid}' should contain '{name}'"

    def test_initial_state_is_empty(self, _mock_streamlit) -> None:
        app = _import_app()
        session = _mock_streamlit.session_state

        app.init_session_state()

        for i in range(4):
            assert session[f"agent_{i}_messages"] == []
            assert session[f"agent_{i}_turn"] == 0
            assert session[f"agent_{i}_metrics"] == []

    def test_idempotent_on_second_call(self, _mock_streamlit) -> None:
        app = _import_app()
        session = _mock_streamlit.session_state

        app.init_session_state()
        first_sids = [session[f"agent_{i}_sid"] for i in range(4)]

        # Second call should not reinitialize
        app.init_session_state()
        second_sids = [session[f"agent_{i}_sid"] for i in range(4)]

        assert first_sids == second_sids

    def test_creates_executor_for_concurrent_requests(self, _mock_streamlit) -> None:
        """Session state should include ThreadPoolExecutor for concurrent HTTP."""
        from concurrent.futures import ThreadPoolExecutor

        app = _import_app()
        session = _mock_streamlit.session_state

        app.init_session_state()

        assert "executor" in session
        assert isinstance(session.executor, ThreadPoolExecutor)


class TestNonStreamResponse:
    """non_stream_response() handles HTTP errors gracefully."""

    def test_http_error_returns_error_message(self) -> None:
        app = _import_app()

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal server error"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            text, metrics = app.non_stream_response(
                [{"role": "user", "content": "test"}], "session_1",
                temperature=0.7, top_p=1.0, max_tokens=512,
            )

        assert "[Error: HTTP 500" in text
        assert metrics == {}

    def test_connection_error_propagates(self) -> None:
        """Network failure should raise (caller handles in render_agent_column)."""
        import httpx

        app = _import_app()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch("httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.ConnectError):
                app.non_stream_response(
                    [{"role": "user", "content": "test"}], "session_1",
                    temperature=0.7, top_p=1.0, max_tokens=512,
                )


class TestCheckServer:
    """check_server() returns None on failure, dict on success."""

    def test_returns_none_on_connection_error(self) -> None:
        import httpx

        app = _import_app()

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        with patch("httpx.Client", return_value=mock_client):
            result = app.check_server()

        assert result is None

    def test_returns_none_on_non_200_health(self) -> None:
        app = _import_app()

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_client):
            result = app.check_server()

        assert result is None

    def test_returns_empty_dict_when_memory_unavailable(self) -> None:
        app = _import_app()

        health_resp = MagicMock()
        health_resp.status_code = 200

        mem_resp = MagicMock()
        mem_resp.status_code = 404

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = [health_resp, mem_resp]

        with patch("httpx.Client", return_value=mock_client):
            result = app.check_server()

        assert result == {}
