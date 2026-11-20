"""Small ZeroMQ helpers for local, bidirectional IPC, i guess that's gonna be funny"""

from pathlib import Path

import zmq

def create_ipc_pair_socket(
    endpoint: str | Path,
    *,
    bind: bool,
    context: zmq.Context | None = None,
    linger_ms: int = 0,
) -> zmq.Socket:
    """Create a full-duplex PAIR socket at a local IPC endpoint.

    Exactly one peer should bind and the other should connect. Pass the same
    endpoint to both processes, e.g. ``/tmp/zoochecker.sock`` or
    ``ipc:///tmp/zoochecker.sock``. Messages are transient; this is not a
    durable queue.
    """
    if isinstance(endpoint, Path) or "://" not in str(endpoint):
        socket_path = Path(endpoint).expanduser().resolve()
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        address = f"ipc://{socket_path}"
    else:
        address = str(endpoint)
        if not address.startswith("ipc://"):
            raise ValueError("endpoint must be a filesystem path or ipc:// URL")

    socket = (context or zmq.Context.instance()).socket(zmq.PAIR)
    socket.setsockopt(zmq.LINGER, linger_ms)
    try:
        if bind:
            socket.bind(address)
        else:
            socket.connect(address)
    except Exception:
        socket.close(linger=0)
        raise
    return socket
