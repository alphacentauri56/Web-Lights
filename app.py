from flask import Flask, render_template, request, jsonify
import threading
import time
import os

import led_driver
import animations

app = Flask(__name__)

# Configuration
LED_COUNT = 274
LED_BRIGHTNESS = float(os.environ.get('LED_BRIGHTNESS', 0.5))
LED_ORDER = None  # led_driver will handle ordering if using real hardware

# Create a pixel driver (wraps real NeoPixel or a fake driver for development)
driver = led_driver.create_pixel_driver(LED_COUNT, brightness=LED_BRIGHTNESS, order=LED_ORDER)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/colours', methods=['POST'])
def colours():
    data = request.json or {}
    colour = data.get('colourdata')
    if not colour:
        return jsonify({"status": "error", "message": "no colour provided"}), 400
    try:
        rgb = led_driver.hex_to_rgb(colour)
    except ValueError:
        return jsonify({"status": "error", "message": "invalid hex colour"}), 400
    driver.fill(rgb)
    driver.show()
    return jsonify({"status": "ok", "message": "colour received", "rgb": rgb})


# Animation control globals
_animation_thread = None
_animation_stop = threading.Event()
_animation_lock = threading.Lock()
_animation_name = None


def _start_animation(target_fn, *args, name=None, **kwargs):
    """Start an animation in a background thread. Stops any running animation first."""
    global _animation_thread, _animation_stop, _animation_name
    _stop_animation()
    _animation_stop.clear()
    _animation_name = name or getattr(target_fn, '__name__', 'animation')

    def runner():
        try:
            target_fn(driver, _animation_stop, *args, **kwargs)
        except Exception as e:
            # keep server alive; logging is useful here
            print('Animation error:', e)

    _animation_thread = threading.Thread(target=runner, daemon=True)
    _animation_thread.start()


def _stop_animation():
    global _animation_thread, _animation_stop, _animation_name
    if _animation_thread and _animation_thread.is_alive():
        _animation_stop.set()
        _animation_thread.join(timeout=2.0)
    _animation_thread = None
    _animation_stop.clear()
    _animation_name = None


@app.route('/animate/start', methods=['POST'])
def animate_start():
    data = request.json or {}
    name = data.get('name', 'rainbow')
    speed = float(data.get('speed', 0.02))
    # Map name to function
    if name == 'rainbow':
        _start_animation(animations.rainbow_cycle, speed, name='rainbow')
    elif name == 'breathing':
        color = data.get('color', '#ffffff')
        try:
            rgb = led_driver.hex_to_rgb(color)
        except ValueError:
            return jsonify({"status": "error", "message": "invalid color"}), 400
        _start_animation(animations.breathing, rgb, speed, name='breathing')
    elif name == 'gradient':
        start = data.get('start', '#000000')
        end = data.get('end', '#ffffff')
        try:
            rgb_start = led_driver.hex_to_rgb(start)
            rgb_end = led_driver.hex_to_rgb(end)
        except ValueError:
            return jsonify({"status": "error", "message": "invalid color"}), 400
        # gradient runs once; use a tiny wrapper
        def once_gradient(drv, stop_event, s, e):
            animations.fill_linear_gradient(drv, s, e, show=True)

        _start_animation(once_gradient, rgb_start, rgb_end, name='gradient')
    else:
        return jsonify({"status": "error", "message": "unknown animation"}), 400

    return jsonify({"status": "ok", "message": f"started {name}"})


@app.route('/animate/stop', methods=['POST'])
def animate_stop():
    _stop_animation()
    # Optionally clear pixels when stopping
    driver.fill((0, 0, 0))
    driver.show()
    return jsonify({"status": "ok", "message": "stopped"})


@app.route('/status', methods=['GET'])
def status():
    running = _animation_thread is not None and _animation_thread.is_alive()
    return jsonify({"running": running, "name": _animation_name})


if __name__ == '__main__':
    # On a Pi you may run with sudo; on dev machines the fake driver is used
    app.run(host='0.0.0.0', port=80, debug=False)