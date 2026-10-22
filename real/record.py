"""Record the real game for calibrating the live agent: the screen (half size, as fast as it comes) and your W / E
presses, with exact times.

    py -m real.record          # X stops; saved in real/recordings/<date time>/

Start it, go to the game full screen, play a Tower run from the drop-in, then press X.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2

from real.screen import Grabber, user32

KEYS = {"W": 0x57, "E": 0x45}
VK_STOP = 0x58  # X (the game doesn't use it; laptops need Fn for F-keys)
OUT = Path(__file__).resolve().parent / "recordings"


def pressed(vk) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def log_keys(events, stop):
    """Every change of W / E, polled once a millisecond: (time, key, down)."""
    state = dict.fromkeys(KEYS, False)
    while not stop.is_set():
        t = time.perf_counter()
        for name, vk in KEYS.items():
            down = pressed(vk)
            if down != state[name]:
                state[name] = down
                events.append((t, name, down))
        time.sleep(0.001)


def main():
    folder = OUT / time.strftime("%Y-%m-%d %H-%M-%S")
    folder.mkdir(parents=True)
    events, frames, stop = [], [], threading.Event()
    threading.Thread(target=log_keys, args=(events, stop), daemon=True).start()
    grabber = Grabber()
    print(f"recording to {folder}. Play a Tower run from the drop-in; X stops.")
    try:
        while not pressed(VK_STOP):
            t0 = time.perf_counter()
            img = grabber.grab()
            t = (t0 + time.perf_counter()) / 2
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(folder / f"{len(frames):05d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(t)
    finally:
        stop.set()
        grabber.close()
        (folder / "log.json").write_text(json.dumps({"scale": 0.5, "frame_times": frames, "keys": events}))
    span = frames[-1] - frames[0] if len(frames) > 1 else 0
    print(f"saved {len(frames)} frames over {span:.1f} s ({len(frames) / max(span, 1e-9):.0f} a second) "
          f"and {len(events)} key changes.")


if __name__ == "__main__":
    main()
