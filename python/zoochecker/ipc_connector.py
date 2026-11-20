"""Small ZeroMQ helpers for local, bidirectional IPC, i guess that's gonna be funny"""

from pathlib import Path

import zmq.asyncio


def create_async_ipc_pair_socket(
    endpoint: str | Path, *, bind: bool
) -> zmq.asyncio.Socket:
    if isinstance(endpoint, Path) or "://" not in str(endpoint):
        socket_path = Path(endpoint).expanduser().resolve()
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        address = f"ipc://{socket_path}"
    else:
        address = str(endpoint)
        if not address.startswith("ipc://"):
            raise ValueError("endpoint must be a filesystem path or ipc:// URL")

    socket = zmq.asyncio.Context.instance().socket(zmq.PAIR)
    try:
        socket.setsockopt(zmq.LINGER, 0)
        if bind:
            socket.bind(address)
        else:
            socket.connect(address)
    except Exception:
        socket.close(linger=0)
        raise
    return socket
