"""Fast screen grabs (Windows GDI), in real pixels, as numpy BGR arrays."""
from __future__ import annotations

import ctypes
from ctypes import wintypes

import numpy as np

user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
user32.SetProcessDPIAware()  # real pixels, not the display-scaled ones

SRCCOPY, DIB_RGB_COLORS = 0x00CC0020, 0

# 64-bit handles: without these, ctypes would cut them to 32-bit ints
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.BitBlt.argtypes = [wintypes.HDC] + [ctypes.c_int] * 4 + [wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                            ctypes.c_void_p, wintypes.UINT]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def screen_size() -> tuple[int, int]:
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


class Grabber:
    """Copies a rectangle of the screen. Reuses its buffers, so keep one per rectangle size."""

    def __init__(self, left=0, top=0, width=None, height=None):
        sw, sh = screen_size()
        self.left, self.top = left, top
        self.w, self.h = width or sw - left, height or sh - top
        self.screen_dc = user32.GetDC(0)
        self.mem_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        self.bitmap = gdi32.CreateCompatibleBitmap(self.screen_dc, self.w, self.h)
        gdi32.SelectObject(self.mem_dc, self.bitmap)
        self.info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), self.w, -self.h, 1, 32, 0, 0, 0, 0, 0, 0)
        self.buf = np.empty((self.h, self.w, 4), np.uint8)

    def grab(self) -> np.ndarray:
        """The rectangle as an (h, w, 3) BGR array (a view into a reused buffer: copy it to keep it)."""
        gdi32.BitBlt(self.mem_dc, 0, 0, self.w, self.h, self.screen_dc, self.left, self.top, SRCCOPY)
        gdi32.GetDIBits(self.mem_dc, self.bitmap, 0, self.h, self.buf.ctypes.data, ctypes.byref(self.info),
                        DIB_RGB_COLORS)
        return self.buf[..., :3]

    def close(self):
        gdi32.DeleteObject(self.bitmap)
        gdi32.DeleteDC(self.mem_dc)
        user32.ReleaseDC(0, self.screen_dc)
