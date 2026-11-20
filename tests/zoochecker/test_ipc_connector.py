import tempfile
import unittest
from pathlib import Path

import zmq

from zoochecker.ipc_connector import create_ipc_pair_socket


class IpcPairSocketTests(unittest.TestCase):
    def _assert_round_trip(self, endpoint: str | Path) -> None:
        server = create_ipc_pair_socket(endpoint, bind=True)
        client = create_ipc_pair_socket(endpoint, bind=False)
        try:
            client.send_json({"type": "request", "value": 42})
            self.assertEqual(
                self._receive_json(server), {"type": "request", "value": 42}
            )

            server.send_json({"type": "response", "value": 42})
            self.assertEqual(
                self._receive_json(client), {"type": "response", "value": 42}
            )
        finally:
            client.close()
            server.close()

    def _receive_json(self, socket: zmq.Socket) -> object:
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        events = dict(poller.poll(timeout=2_000))
        self.assertIn(socket, events, "timed out waiting for IPC message")
        return socket.recv_json()

    def test_path_endpoint_is_full_duplex_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = Path(temp_dir) / "nested" / "zoochecker.sock"
            self._assert_round_trip(endpoint)
            self.assertTrue(endpoint.parent.is_dir())

    def test_ipc_url_endpoint_is_full_duplex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = f"ipc://{Path(temp_dir) / 'zoochecker.sock'}"
            self._assert_round_trip(endpoint)

    def test_rejects_non_ipc_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "ipc://"):
            create_ipc_pair_socket("tcp://127.0.0.1:5555", bind=True)


if __name__ == "__main__":
    unittest.main()
