"""The /search HTTP contract the Search page depends on.

These exist because the query parameters were once dropped from the route
while `SearchService` kept accepting them, and the whole suite still passed:
every test called the service directly, so nothing noticed that the UI's type
filter and paging had quietly stopped reaching it. The filter row simply
vanished from the app. Testing the service is not testing the endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from models.schemas.file_schemas import FileType


@pytest.fixture
def client(isolated_env):
    """Points at a per-test empty index, not the developer's real one.

    Without `isolated_env` these fail the moment OmniFind is running, because
    the embedded Qdrant takes an exclusive lock on its storage folder - a test
    suite that only passes when the app is closed is a test suite people learn
    to ignore. These assert on the HTTP contract, not on retrieved content, so
    an empty index is all they need.
    """
    from main import app

    return TestClient(app)


def test_search_accepts_the_file_type_parameter(client):
    """The type filter chips send this. Without it FastAPI ignores the unknown
    parameter and silently returns unfiltered results."""
    response = client.get("/search", params={"q": "anything", "file_type": "image"})

    assert response.status_code == 200
    assert response.json()["filtered_to"] == FileType.image.value


def test_search_accepts_the_limit_parameter(client):
    """"Explore more" pages through one over-fetched response, so the page asks
    for far more rows than it shows."""
    response = client.get("/search", params={"q": "anything", "limit": 50})

    assert response.status_code == 200


def test_limit_caps_the_number_of_files_returned(client):
    response = client.get("/search", params={"q": "the", "limit": 2})

    assert response.status_code == 200
    assert len(response.json()["results"]) <= 2


def test_every_file_type_the_ui_offers_is_accepted(client):
    """The chips are built from these three values - one the API rejects would
    break that chip and nothing else, which is easy to miss by hand."""
    for file_type in ("document", "image", "code"):
        response = client.get("/search", params={"q": "anything", "file_type": file_type})

        assert response.status_code == 200, f"{file_type} rejected"
        assert response.json()["filtered_to"] == file_type


def test_an_unknown_file_type_is_rejected(client):
    response = client.get("/search", params={"q": "anything", "file_type": "video"})

    assert response.status_code == 422


def test_search_without_optional_parameters_still_works(client):
    """The parameters are additive - older callers must not break."""
    response = client.get("/search", params={"q": "anything"})

    assert response.status_code == 200
    assert response.json()["filtered_to"] is None


def test_searching_a_fresh_install_returns_empty_not_an_error(client):
    """The very first thing a new user does.

    The Qdrant collection is created by indexing, not at startup, so before any
    folder is added it does not exist - and qdrant-client raises rather than
    returning nothing when asked to search a collection that is not there.
    That turned "install, open, type something" into a 500 with a red banner.
    """
    response = client.get("/search", params={"q": "anything at all"})

    assert response.status_code == 200
    assert response.json()["results"] == []
