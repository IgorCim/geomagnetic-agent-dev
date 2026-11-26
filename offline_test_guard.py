"""Pytest-only network guard, enabled explicitly by tools/verify_offline.py."""
import ipaddress
import socket

_originals = {}


def _local(host):
    if isinstance(host, bytes):host=host.decode('ascii')
    if host in (None, '', 'localhost'):return True
    try:return ipaddress.ip_address(host).is_loopback
    except ValueError:return False


def _connect(sock, address):
    if isinstance(address, tuple) and not _local(address[0]):
        raise RuntimeError('Offline pytest blocked external connection: '+str(address[0]))
    return _originals['connect'](sock,address)


def _resolve(host, *args, **kwargs):
    if not _local(host):
        raise RuntimeError('Offline pytest blocked external DNS: '+str(host))
    return _originals['getaddrinfo'](host,*args,**kwargs)


def pytest_configure(config):
    _originals.update(connect=socket.socket.connect,getaddrinfo=socket.getaddrinfo)
    socket.socket.connect=_connect
    socket.getaddrinfo=_resolve


def pytest_unconfigure(config):
    if _originals:
        socket.socket.connect=_originals['connect']
        socket.getaddrinfo=_originals['getaddrinfo']
        _originals.clear()
