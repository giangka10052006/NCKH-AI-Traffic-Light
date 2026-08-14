# Chạy SUMO-GUI trực tiếp
import os
import subprocess
import sys
import time
from pathlib import Path

SUMO_BIN = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
SUMOCFG = r"D:\Project\NNN\sumo\RL.sumocfg"

if not os.path.exists(SUMO_BIN):
    raise FileNotFoundError(f"Không tìm thấy SUMO GUI tại {SUMO_BIN}")

print("Đang mở SUMO GUI với file config:", SUMOCFG)
proc = subprocess.Popen([SUMO_BIN, '-c', SUMOCFG], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

try:
    while True:
        time.sleep(1)
        if proc.poll() is not None:
            print("SUMO GUI đã đóng")
            break
except KeyboardInterrupt:
    print("Đã dừng")
    proc.terminate()
