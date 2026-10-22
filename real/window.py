"""Finding the game's browser window, bringing it to the front, and clicking or pressing keys in it, only ever
while it is the window in front (so a stray click or key can't land in another program)."""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
user32.GetForegroundWindow.restype = wintypes.HWND
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.IsIconic.argtypes = [wintypes.HWND]

GAME_TITLE = "Getaway Shootout"


def title(hwnd) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_game():
    """The browser window with the game in it, or None."""
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd) and title(hwnd).startswith(GAME_TITLE):
            found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found[0] if found else None


def game_in_front() -> bool:
    return title(user32.GetForegroundWindow()).startswith(GAME_TITLE)


def bring_to_front(hwnd) -> bool:
    """Windows only lets the program the user last used change the window in front; a tap of Alt (which the
    game ignores) counts as that."""
    if game_in_front():
        return True
    user32.keybd_event(0x12, 0, 0, 0)  # Alt down
    user32.keybd_event(0x12, 0, 2, 0)  # Alt up
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # restore if minimised
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)
    return game_in_front()


# --- mouse and keys, only into the game -------------------------------------------------------------------

class NotInFront(RuntimeError):
    pass


def _check():
    if not game_in_front():
        raise NotInFront(f"the game isn't the window in front ({title(user32.GetForegroundWindow())!r})")


def click(x: int, y: int):
    """Left click at screen pixel (x, y)."""
    _check()
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.05)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up


# keys the game's menus read: (virtual-key code, scan code, extended key). It needs the scan code.
KEYS = {"left": (0x25, 0x4B, True), "right": (0x27, 0x4D, True), "enter": (0x0D, 0x1C, False),
        "space": (0x20, 0x39, False), "r": (0x52, 0x13, False)}


def tap(name: str, hold: float = 0.08):
    """Press and release one key, by name (KEYS)."""
    _check()
    vk, scan, extended = KEYS[name]
    user32.keybd_event(vk, scan, int(extended), 0)
    time.sleep(hold)
    user32.keybd_event(vk, scan, int(extended) | 2, 0)
