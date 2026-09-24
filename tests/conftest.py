import os

import pytest


@pytest.fixture(autouse=True)
def isolate_registry_state():
    os.makedirs("data", exist_ok=True)
    registry_path = os.path.join("data", "documents.json")

    for filename in list(os.listdir("data")):
        if filename == "documents.json":
            continue
        os.remove(os.path.join("data", filename))

    if os.path.exists(registry_path):
        os.remove(registry_path)

    yield

    for filename in list(os.listdir("data")):
        if filename == "documents.json":
            continue
        os.remove(os.path.join("data", filename))

    if os.path.exists(registry_path):
        os.remove(registry_path)
