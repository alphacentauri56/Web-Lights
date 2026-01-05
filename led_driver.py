"""led_driver.py
Wrapper around NeoPixel with safe fallback for development and color helpers.
"""
import threading
import colorsys
import os

try:
    import board
    import neopixel
    _HAVE_NEOPIXEL = True
except Exception:
    _HAVE_NEOPIXEL = False


def hex_to_rgb(hexstr):
    """Convert '#RRGGBB' or 'RRGGBB' to (r,g,b) tuple of ints 0-255."""
    if not isinstance(hexstr, str):
        raise ValueError('hex string required')
    s = hexstr.lstrip('#')
    if len(s) != 6:
        raise ValueError('invalid hex length')
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))


def hsv_to_rgb255(h, s, v):
    """Convert HSV (0..1) to RGB 0..255"""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class FakeNeoPixel:
    """Simple in-memory fake NeoPixel for development/testing."""
    def __init__(self, n, brightness=0.5, auto_write=False, pixel_order=None):
        self.n = n
        self.brightness = brightness
        self.auto_write = auto_write
        self.pixel_order = pixel_order
        self.data = [(0, 0, 0)] * n

    def __setitem__(self, idx, val):
        # support slices and ints simply
        self.data[idx] = val

    def fill(self, color):
        self.data = [color] * self.n

    def show(self):
        # no-op for fake device; could write to a log or file if desired
        return


class PixelDriver:
    """Driver object exposing basic pixel operations and a show lock."""
    def __init__(self, n, brightness=0.5, order=None):
        if _HAVE_NEOPIXEL:
            # If neopixel is available (likely on a Pi), use that
            pixels = neopixel.NeoPixel(board.D18, n, brightness=brightness, auto_write=False, pixel_order=order)
        else:
            pixels = FakeNeoPixel(n, brightness=brightness, auto_write=False, pixel_order=order)
        self.pixels = pixels
        self.lock = threading.Lock()
        self.n = n

    def __setitem__(self, idx, val):
        self.pixels[idx] = val

    def fill(self, color):
        self.pixels.fill(color)

    def show(self):
        with self.lock:
            self.pixels.show()

    def __len__(self):
        return self.n


def create_pixel_driver(n, brightness=0.5, order=None):
    """Factory: return a PixelDriver wrapping real or fake NeoPixel."""
    return PixelDriver(n, brightness=brightness, order=order)
