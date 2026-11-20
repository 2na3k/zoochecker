import asyncio
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from zoochecker.ipc_connector import create_async_ipc_pair_socket, ipc_endpoints


class IpcPairSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_allocated_endpoints_are_distinct_and_cleaned_up(self) -> None:
        with ipc_endpoints() as (tokenizer, worker), ipc_endpoints() as other:
            self.assertNotEqual(tokenizer, worker)
            self.assertNotEqual(tokenizer.parent, other[0].parent)
            self.assertEqual(tokenizer.parent.stat().st_mode & 0o777, 0o700)
            await self._assert_round_trip(tokenizer)
            await self._assert_round_trip(worker)
        self.assertFalse(tokenizer.parent.exists())
        self.assertFalse(other[0].parent.exists())

    async def _assert_round_trip(self, endpoint: str | Path) -> None:
        with (
            closing(create_async_ipc_pair_socket(endpoint, bind=True)) as server,
            closing(create_async_ipc_pair_socket(endpoint, bind=False)) as client,
        ):
            async with asyncio.timeout(2):
                await client.send_json({"type": "request", "value": 42})
                self.assertEqual(
                    await server.recv_json(), {"type": "request", "value": 42}
                )
                await server.send_json({"type": "response", "value": 42})
                self.assertEqual(
                    await client.recv_json(), {"type": "response", "value": 42}
                )

    async def test_path_endpoint_is_full_duplex_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = Path(temp_dir) / "nested" / "zoochecker.sock"
            await self._assert_round_trip(endpoint)
            self.assertTrue(endpoint.parent.is_dir())

    async def test_ipc_url_endpoint_is_full_duplex(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            endpoint = f"ipc://{Path(temp_dir) / 'zoochecker.sock'}"
            await self._assert_round_trip(endpoint)

    def test_rejects_non_ipc_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "ipc://"):
            create_async_ipc_pair_socket("tcp://127.0.0.1:5555", bind=True)


if __name__ == "__main__":
    unittest.main()
