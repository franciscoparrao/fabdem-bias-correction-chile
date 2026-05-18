#!/usr/bin/env python3
"""E1a orchestrator.

Runs the 7 pipeline stages sequentially as subprocesses (RAM freed between),
with:
  - Checkpoint markers (status/*.done)
  - psutil watchdog (kills if RAM > MAX_GB)
  - Per-stage log to logs/stage_NN.log

Usage:
  python3 run_e1a.py            # full run
  python3 run_e1a.py --reset    # clear all checkpoints first
"""
import os
import sys
import time
import json
import signal
import subprocess
import threading
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "status"
LOGS = ROOT / "logs"
STATUS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

MAX_RAM_GB = 8.0
WATCHDOG_INTERVAL_S = 5

STAGES = [
    ("01_fabdem",   ["python3", "steps/01_fabdem.py"]),
    ("02_terrain",  ["bash",    "steps/02_terrain.sh"]),
    ("03_satellite",["python3", "steps/03_satellite.py"]),
    ("04_atl08",    ["python3", "steps/04_atl08.py"]),
    ("05_extract",  ["python3", "steps/05_extract.py"]),
    ("06_sample",   ["python3", "steps/06_sample.py"]),
    ("07_train",    ["python3", "steps/07_train.py"]),
]

class Watchdog:
    """Background thread that watches RAM of a target subprocess + descendants."""
    def __init__(self, max_gb, interval_s=5):
        self.max_gb = max_gb
        self.interval = interval_s
        self._stop = threading.Event()
        self._target = None
        self._max_seen_per_stage = {}
        self._current_stage = None
        self._t = None

    def watch(self, popen, stage_name):
        self._target = popen
        self._current_stage = stage_name
        self._max_seen_per_stage.setdefault(stage_name, 0.0)

    def start(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()

    def _measure(self):
        if not self._target:
            return 0.0
        try:
            p = psutil.Process(self._target.pid)
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            return rss / 1e9
        except psutil.NoSuchProcess:
            return 0.0

    def _loop(self):
        while not self._stop.is_set():
            if self._target and self._target.poll() is None:
                gb = self._measure()
                stage = self._current_stage
                if gb > self._max_seen_per_stage.get(stage, 0):
                    self._max_seen_per_stage[stage] = gb
                if gb > self.max_gb:
                    print(f"\n[WATCHDOG] stage={stage} RAM={gb:.2f}GB > {self.max_gb}GB — KILLING")
                    try:
                        # Kill the entire process tree
                        p = psutil.Process(self._target.pid)
                        for c in p.children(recursive=True):
                            try: c.kill()
                            except: pass
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                    return
            self._stop.wait(self.interval)

    def report(self):
        return dict(self._max_seen_per_stage)


def run_stage(name, cmd, watchdog):
    marker = STATUS / f"{name}.done"
    if marker.exists():
        print(f"✓ {name} (already done, skipping)")
        return True
    log_path = LOGS / f"{name}.log"
    print(f"\n▶ {name}  (log → {log_path.name})")
    start = time.time()
    with open(log_path, "w", buffering=1) as logf:
        popen = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        watchdog.watch(popen, name)
        rc = popen.wait()
    elapsed = time.time() - start
    peak = watchdog._max_seen_per_stage.get(name, 0.0)
    print(f"  rc={rc}  time={elapsed:.1f}s  peak_RAM={peak:.2f}GB")

    # Show last log lines for status visibility
    try:
        lines = log_path.read_text().splitlines()
        for line in lines[-10:]:
            print(f"  | {line}")
    except Exception:
        pass

    if rc == 0:
        marker.write_text(json.dumps({
            "elapsed_s": round(elapsed, 2),
            "peak_ram_gb": round(peak, 3),
            "cmd": cmd,
        }))
        return True
    print(f"✗ {name} FAILED (rc={rc})")
    return False


def main():
    if "--reset" in sys.argv:
        for m in STATUS.glob("*.done"):
            m.unlink()
        print("Reset all checkpoints.")

    print(f"=== E1a pipeline ===")
    print(f"Workdir: {ROOT}")
    print(f"Max RAM: {MAX_RAM_GB} GB  (watchdog every {WATCHDOG_INTERVAL_S}s)")
    print(f"Stages: {[s[0] for s in STAGES]}")

    wd = Watchdog(max_gb=MAX_RAM_GB, interval_s=WATCHDOG_INTERVAL_S)
    wd.start()
    pipeline_start = time.time()

    try:
        for name, cmd in STAGES:
            ok = run_stage(name, cmd, wd)
            if not ok:
                print(f"\n✗ Pipeline aborted at {name}")
                return 1
    finally:
        wd.stop()
        report = {
            "pipeline_elapsed_s": round(time.time() - pipeline_start, 1),
            "max_ram_per_stage_gb": wd.report(),
        }
        (LOGS / "pipeline_summary.json").write_text(json.dumps(report, indent=2))
        print(f"\n=== Summary ===")
        print(json.dumps(report, indent=2))

    print(f"\n✅ E1a complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
