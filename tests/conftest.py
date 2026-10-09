import os

import pytest


@pytest.fixture(autouse=True)
def isolate_registry_state(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    yield
