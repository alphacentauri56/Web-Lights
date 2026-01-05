"""animations.py
Simple animations and gradient generators for the pixel driver.
Functions accept (driver, stop_event, ...) where driver is PixelDriver from led_driver.
"""
import time
import math
import led_driver


def fill_linear_gradient(driver, start_color, end_color, show=True):
    """Fill the strip with a linear gradient from start_color to end_color.

    start_color and end_color are (r,g,b) tuples.
    """
    n = len(driver)
    if n <= 1:
        driver.fill(start_color)
        if show:
            driver.show()
        return

    for i in range(n):
        t = i / (n - 1)
        r = int(start_color[0] + (end_color[0] - start_color[0]) * t)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * t)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * t)
        driver[i] = (r, g, b)
    if show:
        driver.show()


def rainbow_cycle(driver, stop_event, speed=0.02):
    """Continuous rainbow cycle. Call with stop_event to stop."""
    n = len(driver)
    pos = 0.0
    while not stop_event.is_set():
        for i in range(n):
            h = ((i / n) + pos) % 1.0
            color = led_driver.hsv_to_rgb255(h, 1.0, 1.0)
            driver[i] = color
        driver.show()
        pos = (pos + 0.005) % 1.0
        time.sleep(speed)


def breathing(driver, stop_event, color=(255, 255, 255), speed=0.02):
    """Breathing effect using a single color whose brightness oscillates."""
    n = len(driver)
    t = 0.0
    while not stop_event.is_set():
        brightness = (math.sin(t) + 1.0) / 2.0  # 0..1
        for i in range(n):
            r = int(color[0] * brightness)
            g = int(color[1] * brightness)
            b = int(color[2] * brightness)
            driver[i] = (r, g, b)
        driver.show()
        t += 0.1
        time.sleep(speed)


def theater_chase(driver, stop_event, color=(255, 0, 0), speed=0.1, spacing=3):
    """A simple theater chase: a moving dot pattern."""
    n = len(driver)
    offset = 0
    while not stop_event.is_set():
        for i in range(n):
            if (i + offset) % spacing == 0:
                driver[i] = color
            else:
                driver[i] = (0, 0, 0)
        driver.show()
        offset = (offset + 1) % spacing
        time.sleep(speed)
