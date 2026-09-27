"""Flash an image over USB alone: flash.py loader.bin fw.bin. The ROM DFU bootloader writes RAM correctly but not
flash on this board, so it only loads loader.bin into RAM, checked by reading it back, and starts it. The loader
erases and programs flash and checks every 4 KB chunk and the whole image."""
import struct
import sys
import time
import zlib

import usb.core

VID, DFU_PID, LOADER_PID, BENCH_PID = 0x0483, 0xDF11, 0x5741, 0x5740
LOADER_AT, BASE, CHUNK = 0x20008000, 0x08000000, 4096
SECTORS = [BASE + o for o in (0, 0x4000, 0x8000, 0xC000, 0x10000, 0x20000, 0x40000, 0x60000, 0x80000)]


def open_device(pid, wait=60):
    t0, asked = time.time(), False
    while time.time() - t0 < wait or asked:
        d = usb.core.find(idVendor=VID, idProduct=pid)
        if d is not None:
            try:
                d.set_configuration()
                return d
            except usb.core.USBError as e:
                if e.errno != 13:
                    raise
                if not asked:
                    print(f"no access to /dev/bus/usb/{d.bus:03d}/{d.address:03d}", flush=True)
                    asked = True
        time.sleep(0.5)
    sys.exit(f"no device {VID:04x}:{pid:04x}")


def dfu_run_from_ram(d, image):
    status = lambda: d.ctrl_transfer(0xA1, 3, 0, 0, 6, 5000)[4]

    def settle():
        while status() == 4:
            pass

    def idle():
        s = status()
        if s == 10:
            d.ctrl_transfer(0x21, 4, 0, 0, None, 5000)
        elif s != 2:
            d.ctrl_transfer(0x21, 6, 0, 0, None, 5000)

    def set_address(a):
        idle()
        d.ctrl_transfer(0x21, 1, 0, 0, struct.pack("<BI", 0x21, a), 5000)
        settle()

    def read(o, n):
        set_address(LOADER_AT + o)
        idle()
        return bytes(d.ctrl_transfer(0xA1, 2, 2, 0, n, 5000))

    # a lost reply can make the bootloader take a command as data: redo every chunk until all of them read back
    chunks = list(range(0, len(image), 1024))
    for attempt in range(20):
        for o in chunks:
            try:
                set_address(LOADER_AT + o)
                d.ctrl_transfer(0x21, 1, 2, 0, image[o:o + 1024], 5000)
                settle()
            except usb.core.USBError:
                time.sleep(0.2)
        bad = []
        for o in range(0, len(image), 1024):
            try:
                if read(o, len(image[o:o + 1024])) != image[o:o + 1024]:
                    bad.append(o)
            except usb.core.USBError:
                bad.append(o)
                time.sleep(0.2)
        if not bad:
            break
        chunks = bad
    else:
        sys.exit("the loader does not read back from RAM")
    print(f"loader in RAM after {attempt + 1} passes", flush=True)
    set_address(LOADER_AT)
    d.ctrl_transfer(0x21, 1, 0, 0, None, 5000)
    try:
        status()
    except usb.core.USBError:
        pass


def request(d, cmd, addr=0, data=b"", n=0):
    d.write(0x01, struct.pack("<4I", ord(cmd), addr, len(data) or n, zlib.crc32(data)) + data, 10000)
    return struct.unpack("<2I", bytes(d.read(0x81, 64, 10000)))


loader, image = (open(p, "rb").read() for p in sys.argv[1:3])
image += b"\xff" * (-len(image) % 4)
if usb.core.find(idVendor=VID, idProduct=LOADER_PID) is None:
    bench = usb.core.find(idVendor=VID, idProduct=BENCH_PID)
    if bench is not None and usb.core.find(idVendor=VID, idProduct=DFU_PID) is None:
        # the benchmark reboots into the ROM bootloader on 'b'
        for i in (0, 1):
            if bench.is_kernel_driver_active(i):
                bench.detach_kernel_driver(i)
        bench.write(0x01, b"b", 5000)
        time.sleep(2)
    dfu_run_from_ram(open_device(DFU_PID), loader)
d = open_device(LOADER_PID)
_, ident = request(d, "I")
print(f"device {ident & 0xFFF:#x}, {ident >> 16} KB flash", flush=True)
for i, s in enumerate(SECTORS[:-1]):
    if s < BASE + len(image):
        status, _ = request(d, "E", i)
        if status:
            sys.exit(f"erase sector {i}: flash status {status:#x}")
retries = 0
for o in range(0, len(image), CHUNK):
    for attempt in range(5):
        status, _ = request(d, "W", BASE + o, image[o:o + CHUNK])
        if status == 0:
            break
        retries += 1
    else:
        sys.exit(f"chunk {o:#x}: status {status}")
_, crc = request(d, "C", BASE, n=len(image))
if crc != zlib.crc32(image):
    sys.exit("image CRC differs")
print(f"{len(image)} bytes written and checked, {retries} chunks resent", flush=True)
try:
    request(d, "G")
except usb.core.USBError:
    pass
print("started", flush=True)
