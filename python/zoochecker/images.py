import asyncio
import base64
import binascii
import ipaddress
import socket
from collections.abc import Callable
from io import BytesIO
from typing import Literal

import httpx
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from zoochecker.server_args import ServerArgs
from zoochecker.types import InvalidPrompt

ImageSize = Callable[[int, int], tuple[int, int]]


class ImageURL(BaseModel):
    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] = "auto"


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


async def prepare_chat_images(
    messages: list[dict], resize: ImageSize, args: ServerArgs
) -> list[dict]:
    sources = [
        part["image_url"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    if not sources:
        return messages
    if len(sources) > args.max_images:
        raise InvalidPrompt(
            f"At most {args.max_images} images are supported per request"
        )
    await asyncio.to_thread(_prepare_images, sources, resize, args)
    return messages


def _prepare_images(sources: list[dict], resize: ImageSize, args: ServerArgs) -> None:
    with httpx.Client(trust_env=False, timeout=args.image_timeout_seconds) as client:
        for source in sources:
            try:
                url = source["url"]
                if url.startswith("data:"):
                    media, _, encoded = url.partition(",")
                    if not media.startswith("data:image/") or not media.endswith(
                        ";base64"
                    ):
                        raise InvalidPrompt("Expected a base64 image data URL")
                    if len(encoded) > 4 * ((args.max_image_bytes + 2) // 3):
                        raise InvalidPrompt("Image exceeds the byte limit")
                    data = base64.b64decode(encoded, validate=True)
                else:
                    data = _download_image(client, url, args.max_image_bytes)
                if len(data) > args.max_image_bytes:
                    raise InvalidPrompt("Image exceeds the byte limit")
                with Image.open(BytesIO(data)) as image:
                    if image.width * image.height > args.max_image_pixels:
                        raise InvalidPrompt("Image exceeds the pixel limit")
                    oriented = ImageOps.exif_transpose(image).convert("RGB")
                size = resize(*oriented.size)
                if size != oriented.size:
                    oriented = oriented.resize(size, Image.Resampling.BICUBIC)
                buffer = BytesIO()
                oriented.save(buffer, format="PNG")
                source["url"] = "data:image/png;base64," + base64.b64encode(
                    buffer.getvalue()
                ).decode("ascii")
            except InvalidPrompt:
                raise
            except (
                OSError,
                ValueError,
                binascii.Error,
                httpx.HTTPError,
                Image.DecompressionBombError,
            ) as exc:
                raise InvalidPrompt("Could not read image") from exc


def decode_prepared_images(sources: tuple[str, ...]) -> list[Image.Image]:
    images = []
    for source in sources:
        prefix, _, encoded = source.partition(",")
        if prefix != "data:image/png;base64":
            raise InvalidPrompt("Expected a prepared PNG image")
        with Image.open(
            BytesIO(base64.b64decode(encoded, validate=True)), formats=("PNG",)
        ) as image:
            image.load()
        images.append(image)
    return images


def _download_image(client: httpx.Client, source: str, limit: int) -> bytes:
    url = httpx.URL(source)
    with client.stream(
        "GET",
        _public_address(url),
        headers={
            "Host": url.netloc.decode("ascii"),
            "Accept-Encoding": "identity",
            "Connection": "close",
        },
        extensions={"sni_hostname": url.raw_host.decode("ascii")},
    ) as response:
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_raw():
            if len(data) + len(chunk) > limit:
                raise InvalidPrompt("Image exceeds the byte limit")
            data.extend(chunk)
    return bytes(data)


def _public_address(url: httpx.URL) -> httpx.URL:
    if url.scheme not in ("http", "https") or not url.host or url.userinfo:
        raise InvalidPrompt("Image URLs must use HTTP, HTTPS, or an image data URL")
    addresses = socket.getaddrinfo(
        url.raw_host.decode("ascii"), url.port, type=socket.SOCK_STREAM
    )
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if isinstance(ip, ipaddress.IPv6Address) and (ip.sixtofour or ip.teredo):
            continue
        if ip.is_global and not ip.is_multicast and not ip.is_reserved:
            return url.copy_with(host=str(ip))
    raise InvalidPrompt("Image URL must resolve to a public address")
