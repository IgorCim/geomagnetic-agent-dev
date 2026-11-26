"""The offline harness must fail external access, not report live success."""
import socket
import pytest
import offline_test_guard as guard


def test_offline_dns_blocks_external_hosts_but_keeps_loopback():
    configured=bool(guard._originals)
    if not configured:guard.pytest_configure(None)
    try:
        with pytest.raises(RuntimeError,match='Offline pytest blocked external DNS'):
            socket.getaddrinfo('example.com',443)
        assert socket.getaddrinfo('127.0.0.1',80)
    finally:
        if not configured:guard.pytest_unconfigure(None)


def test_offline_socket_blocks_external_ip_before_connect():
    configured=bool(guard._originals)
    if not configured:guard.pytest_configure(None)
    try:
        with socket.socket() as client:
            with pytest.raises(RuntimeError,match='Offline pytest blocked external connection'):
                client.connect(('192.0.2.1',443))
    finally:
        if not configured:guard.pytest_unconfigure(None)
