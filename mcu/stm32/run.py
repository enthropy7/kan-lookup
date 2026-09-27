"""Start the benchmark and collect its USB serial output until DONE: run.py --out results.txt"""
import argparse
import time

import usb.core

a = argparse.ArgumentParser()
a.add_argument("--out", required=True)
a.add_argument("--specs", help="time these specs, one per line, sent over USB, instead of the built-in run")
a = a.parse_args()
while (d := usb.core.find(idVendor=0x0483, idProduct=0x5740)) is None:
    time.sleep(0.5)
for i in (0, 1):
    try:
        if d.is_kernel_driver_active(i):
            d.detach_kernel_driver(i)
    except usb.core.USBError as e:
        if e.errno == 13:
            raise SystemExit(f"no access to /dev/bus/usb/{d.bus:03d}/{d.address:03d}")
        raise
d.write(0x01, b"p", 5000)
try:
    pong = bytes(d.read(0x82, 64, 3000))
except usb.core.USBTimeoutError:
    raise SystemExit("no answer to ping: the board is busy; it restarts 30 s after nobody reads it")
print(pong.decode(errors="replace").strip(), flush=True)
if a.specs:
    lines = [l for l in open(a.specs).read().split() if l]
    rest = b""
    with open(a.out, "w") as f:
        for n, spec in enumerate(lines):
            d.write(0x01, (spec + "\n").encode(), 20000)
            while True:
                try:
                    rest += bytes(d.read(0x82, 64, 60000))
                except usb.core.USBTimeoutError:
                    raise SystemExit(f"no answer to {spec[:60]}")
                *got, rest = rest.split(b"\n")
                got = [g.decode(errors="replace").strip() for g in got]
                for g in got:
                    if g.startswith("fault"):
                        raise SystemExit(f"{spec[:60]}: {g}")
                    if g.startswith("spec"):
                        f.write(g + "\n")
                f.flush()
                if "sgrid end" in got:
                    break
            if n % 20 == 0:
                print(n, "/", len(lines), flush=True)
    raise SystemExit(print(len(lines), "specs"))
d.write(0x01, b"g", 5000)
t0, n, rest = time.time(), 0, b""
with open(a.out, "w") as f:
    while time.time() - t0 < 4 * 3600:
        try:
            rest += bytes(d.read(0x82, 64, 5000))
        except usb.core.USBTimeoutError:
            continue
        *lines, rest = rest.split(b"\n")
        for line in lines:
            line = line.decode(errors="replace").strip()
            if line.startswith(("board", "spec", "DONE")):
                f.write(line + "\n")
                f.flush()
                n += 1
                if n % 100 == 0 or not line.startswith("spec"):
                    print(line, flush=True)
            if line == "DONE":
                raise SystemExit(print(n, "lines"))
