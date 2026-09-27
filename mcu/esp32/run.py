"""Flash the benchmark and collect its UART output until DONE.
    nix develop --command python run.py [--port /dev/ttyUSB0] --out results.txt"""
import argparse
import subprocess
import time

import serial

a = argparse.ArgumentParser()
a.add_argument("--port", default="/dev/ttyUSB0")
a.add_argument("--out", required=True)
a.add_argument("--no-flash", action="store_true")
a = a.parse_args()
if not a.no_flash:
    subprocess.run(["idf.py", "-p", a.port, "-b", "460800", "flash"], check=True)
with serial.Serial(a.port, 115200, timeout=1) as s, open(a.out, "w") as f:
    s.dtr, s.rts = False, True
    time.sleep(0.1)
    s.rts = False
    t0, n = time.time(), 0
    while time.time() - t0 < 7200:
        line = s.readline().decode(errors="replace").strip()
        if not line:
            continue
        if line.startswith(("board", "spec", "DONE")):
            f.write(line + "\n")
            f.flush()
            n += 1
            if n % 50 == 0 or not line.startswith("spec"):
                print(line, flush=True)
        if line == "DONE":
            break
print(n, "lines")
