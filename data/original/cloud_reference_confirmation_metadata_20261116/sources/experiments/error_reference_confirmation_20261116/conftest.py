"""Unit tests may use tiny random CPU fixtures, never downloads or cloud calls."""

import socket

import pytest


@pytest.fixture(autouse=True)
def prohibit_network_in_confirmation_tests(monkeypatch):
    """Make an accidental pretrained download or service call fail immediately."""

    def blocked(*_args, **_kwargs):
        raise RuntimeError("Network access is forbidden in confirmation unit tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
