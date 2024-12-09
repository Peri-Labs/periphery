import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from queue import Queue
from pydantic import BaseModel
import io
import numpy as np
from periphery.distributed.http_server.server import Server

from tests.unit.mock import MockNode, MockTaskManager

@pytest.fixture
def client():
    node = MockNode()
    server = Server(node)
    return TestClient(server.app)

def test_model_assign(client):
    mock_file = io.BytesIO(b"mock data")
    files = {"file": ("tests/unit/mock_files/mock.onnx", mock_file, "application/octet-stream")}
    response = client.post("/model_assign", files=files)
    assert response.status_code == 200
    assert "message" in response.json()

def test_child_assign(client):
    payload = {
        "outputs": ["1", "2", "3"],
        "host_ip": "192.168.1.1"
    }
    response = client.post("/child_assign", json=payload)
    assert response.status_code == 200

def test_submit_input(client):
    mock_file = io.BytesIO()
    np.savez(mock_file, data=np.array([1, 2, 3]))
    mock_file.seek(0)
    files = {"file": ("input_data.npz", mock_file, "application/octet-stream")}
    response = client.post("/submit_input/1", files=files)
    assert response.status_code == 200

def test_register_node(client):
    payload = {"ip": "192.168.1.2"}
    response = client.post("/register_node", json=payload)
    assert response.status_code == 200
    assert response.json()["message"] == "Node registered successfully"

def test_get_inputs(client):
    response = client.get("/inputs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_output(client):
    assert True
