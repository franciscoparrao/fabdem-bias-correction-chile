#!/usr/bin/env python3
"""Orchestrator: infer 6 tiles + mosaic. Watchdog + checkpoints."""
import os, sys, time, json, subprocess, threading
from pathlib import Path
import psutil

ROOT = Path("/home/franciscoparrao/proyectos/super-resolution-dem")
SCALE_P1 = ROOT / "scale_p1"
SCALE_E1A = ROOT / "scale_e1a"
INFER = SCALE_P1 / "inference"
INFER.mkdir(exist_ok=True)
INFER_SCRIPT = INFER / "infer_tile.py"

MAX_RAM_GB = 8.0

# 6 tiles: 5 in scale_p1/tiles/<name>/, 1 in scale_e1a/
TILES = [
    ("S35W072", SCALE_P1 / "tiles" / "S35W072"),
    ("S35W071", SCALE_P1 / "tiles" / "S35W071"),
    ("S36W071", SCALE_P1 / "tiles" / "S36W071"),
    ("S37W072", SCALE_P1 / "tiles" / "S37W072"),
    ("S37W071", SCALE_P1 / "tiles" / "S37W071"),
    ("S36W072", SCALE_E1A),  # reuse E1a
]


class Watchdog:
    def __init__(self, max_gb): self.max_gb = max_gb; self._target=None; self._max=0; self._stop=threading.Event(); self._stage=None
    def watch(self, p, s): self._target=p; self._stage=s
    def start(self): threading.Thread(target=self._loop, daemon=True).start()
    def stop(self): self._stop.set()
    def _loop(self):
        while not self._stop.is_set():
            if self._target and self._target.poll() is None:
                try:
                    p=psutil.Process(self._target.pid); rss=p.memory_info().rss
                    for c in p.children(recursive=True):
                        try: rss+=c.memory_info().rss
                        except: pass
                    gb=rss/1e9
                    if gb>self._max: self._max=gb
                    if gb>self.max_gb:
                        print(f"\n[WATCHDOG] stage={self._stage} RAM={gb:.2f}GB > {self.max_gb} — KILL")
                        try:
                            for c in p.children(recursive=True): c.kill()
                            p.kill()
                        except: pass
                        return
                except: pass
            self._stop.wait(5)


def run_infer_tile(tile_name, tile_dir, wd):
    print(f"\n{'='*60}\n  INFER {tile_name}\n{'='*60}")
    out_corr = INFER / f"{tile_name}_corrected.tif"
    if out_corr.exists() and out_corr.stat().st_size > 1_000_000:
        print(f"  ✓ already done, skip")
        return True
    log = INFER / f"{tile_name}_infer.log"
    with open(log, "w", buffering=1) as f:
        p = subprocess.Popen(
            ["python3", str(INFER_SCRIPT), str(tile_dir)],
            stdout=f, stderr=subprocess.STDOUT,
        )
        wd.watch(p, tile_name)
        rc = p.wait()
    print(f"  rc={rc}  peak_RAM={wd._max:.2f}GB")
    # Show last 12 log lines
    try:
        for line in log.read_text().splitlines()[-12:]:
            print(f"    | {line}")
    except Exception:
        pass
    return rc == 0


def mosaic_tiles():
    out = INFER / "mm_corrected.tif"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"\n✓ mosaic already exists: {out}")
        return out
    print(f"\n{'='*60}\n  MOSAIC 6 tiles\n{'='*60}")
    inputs = [INFER / f"{t}_corrected.tif" for t, _ in TILES]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        print(f"  ⚠ missing: {missing}")
        return None
    cmd = ["surtgis", "mosaic"] + sum([["-i", str(p)] for p in inputs], []) + \
          ["--compress", str(out)]
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"  → {out} ({out.stat().st_size/1024/1024:.1f} MB)")
    return out if rc == 0 else None


def main():
    print(f"P2 part 1: Raster-wide inference  (max RAM {MAX_RAM_GB} GB)")
    wd = Watchdog(MAX_RAM_GB)
    wd.start()
    t0 = time.time()
    failed = []
    for tile_name, tile_dir in TILES:
        wd._max = 0  # reset per-tile peak
        ok = run_infer_tile(tile_name, tile_dir, wd)
        if not ok:
            failed.append(tile_name)
    if failed:
        print(f"\n⚠ Failed tiles: {failed}")
        return 1
    mosaic_tiles()
    print(f"\n✅ Total elapsed: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
