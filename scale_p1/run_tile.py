#!/usr/bin/env python3
"""Run pipeline stages 01-06 for a single tile.

Usage: python3 run_tile.py <tile_name> <bbox>
  where bbox = "W,S,E,N"
"""
import os, sys, time, json, subprocess, threading
from pathlib import Path
import psutil

SCALE_P1 = Path(__file__).resolve().parent
STEPS = SCALE_P1 / "steps"

MAX_RAM_GB = 8.0
WATCHDOG_INTERVAL_S = 5

STAGES = [
    ("01_fabdem",   ["python3", str(STEPS / "01_fabdem.py")]),
    ("02_terrain",  ["bash",    str(STEPS / "02_terrain.sh")]),
    ("03_satellite",["python3", str(STEPS / "03_satellite.py")]),
    ("04_atl08",    ["python3", str(STEPS / "04_atl08.py")]),
    ("05_extract",  ["python3", str(STEPS / "05_extract.py")]),
    ("06_sample",   ["python3", str(STEPS / "06_sample.py")]),
]


class Watchdog:
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
                try: rss += c.memory_info().rss
                except psutil.NoSuchProcess: pass
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
                    print(f"\n[WATCHDOG] stage={stage} RAM={gb:.2f}GB > {self.max_gb}GB — KILL")
                    try:
                        p = psutil.Process(self._target.pid)
                        for c in p.children(recursive=True):
                            try: c.kill()
                            except: pass
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                    return
            self._stop.wait(self.interval)


def run_stage(name, cmd, workdir, env, wd, log_dir):
    marker = workdir / "status" / f"{name}.done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        print(f"  ✓ {name} (skip, done)")
        return True
    log_path = log_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ▶ {name}  (log → {log_path.name})")
    start = time.time()
    with open(log_path, "w", buffering=1) as logf:
        popen = subprocess.Popen(
            cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wd.watch(popen, name)
        rc = popen.wait()
    elapsed = time.time() - start
    peak = wd._max_seen_per_stage.get(name, 0.0)
    print(f"    rc={rc}  time={elapsed:.1f}s  peak_RAM={peak:.2f}GB")
    if rc == 0:
        marker.write_text(json.dumps({"elapsed_s": round(elapsed, 2),
                                       "peak_ram_gb": round(peak, 3)}))
        return True
    print(f"  ✗ {name} FAILED")
    try:
        for line in log_path.read_text().splitlines()[-15:]:
            print(f"    | {line}")
    except Exception:
        pass
    return False


def main():
    if len(sys.argv) < 3:
        print("Usage: run_tile.py <tile_name> <bbox>")
        sys.exit(1)
    tile_name = sys.argv[1]
    bbox_str = sys.argv[2]
    workdir = SCALE_P1 / "tiles" / tile_name
    log_dir = workdir / "logs"
    workdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PIPELINE_BBOX"] = bbox_str
    env["PIPELINE_WORKDIR"] = str(workdir)

    print(f"\n===== TILE {tile_name}  BBOX={bbox_str} =====")
    print(f"Workdir: {workdir}")

    wd = Watchdog(MAX_RAM_GB, WATCHDOG_INTERVAL_S)
    wd.start()
    t_start = time.time()
    try:
        for name, cmd in STAGES:
            ok = run_stage(name, cmd, workdir, env, wd, log_dir)
            if not ok:
                print(f"\n✗ Tile {tile_name} ABORTED at {name}")
                return 2
    finally:
        wd.stop()

    summary = {
        "tile": tile_name,
        "bbox": bbox_str,
        "elapsed_s": round(time.time() - t_start, 1),
        "peak_ram_per_stage": wd._max_seen_per_stage,
    }
    (workdir / "logs" / "tile_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅ Tile {tile_name} complete in {summary['elapsed_s']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
