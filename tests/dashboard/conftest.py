from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.runtime.caching import cache_utils


@pytest.fixture
def cache_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    """Streamlit's expiry clock, set through `return_value`. Cleared on both sides,
    since each cache keeps the timer it was built with."""
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(cache_utils, "TTLCACHE_TIMER", clock)
    st.cache_data.clear()
    yield clock
    st.cache_data.clear()
