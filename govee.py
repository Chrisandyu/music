# control govee h7015 lights over bluetooth
# python govee.py scan | on | off | color 255 0 0 | brightness 50
# close the govee app first, lights only take one connection

import argparse
import asyncio
from contextlib import asynccontextmanager

from bleak import BleakClient, BleakScanner

ADDRESS = "CE:1C:40:86:5B:15"
CHAR_UUID = "00010203-0405-0607-0809-0a0b0c0d2b11"


# ---------- packets ----------
# 20 bytes: data zero padded to 19 + xor checksum

def packet(*data: int) -> bytes:
    body = list(data) + [0] * (19 - len(data))
    checksum = 0
    for byte in body:
        checksum ^= byte
    return bytes(body + [checksum])


KEEP_ALIVE = packet(0xAA, 0x01)


def power(on: bool) -> bytes:
    return packet(0x33, 0x01, int(on))


def brightness(percent: int) -> bytes:
    if not 1 <= percent <= 100:
        raise ValueError("brightness must be 1-100")
    return packet(0x33, 0x04, percent)


BULBS = 15
ALL_BULBS = (1 << BULBS) - 1


def color(r: int, g: int, b: int, mask: int = ALL_BULBS) -> bytes:
    # 5 zeros after rgb then 2 byte bulb mask, bit i = bulb i, low byte first
    return packet(0x33, 0x05, 0x15, 0x01, r, g, b, 0, 0, 0, 0, 0, mask & 0xFF, mask >> 8)


# ---------- bluetooth ----------

@asynccontextmanager
async def connect(address: str):
    # scan first so bluez treats it as bluetooth le (else br-connection-canceled)
    device = await BleakScanner.find_device_by_address(address, timeout=10)
    if device is None:
        raise RuntimeError(f"{address} not found. Close the Govee app on your phone and move closer.")
    async with BleakClient(device, timeout=20) as client:
        await client.start_notify(CHAR_UUID, lambda _, __: None)
        yield client


async def send(client: BleakClient, data: bytes) -> None:
    await client.write_gatt_char(CHAR_UUID, data, response=False)


# ---------- commands ----------

async def scan() -> None:
    devices = await BleakScanner.discover(timeout=8)
    for d in devices:
        if d.name and "govee" in d.name.lower():
            print(f"{d.address}  {d.name}")


async def send_one(address: str, data: bytes) -> None:
    async with connect(address) as client:
        await send(client, data)
        await asyncio.sleep(1)  # let the write go out before disconnecting


def main() -> None:
    parser = argparse.ArgumentParser(description="Control Govee H7015 lights")
    parser.add_argument("--address", default=ADDRESS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan")
    sub.add_parser("on")
    sub.add_parser("off")
    p_color = sub.add_parser("color")
    p_color.add_argument("rgb", nargs=3, type=int, metavar="0-255")
    p_bright = sub.add_parser("brightness")
    p_bright.add_argument("percent", type=int)
    args = parser.parse_args()

    if args.command == "scan":
        asyncio.run(scan())
        return
    data = {
        "on": lambda: power(True),
        "off": lambda: power(False),
        "color": lambda: color(*args.rgb),
        "brightness": lambda: brightness(args.percent),
    }[args.command]()
    asyncio.run(send_one(args.address, data))


if __name__ == "__main__":
    main()
