import os
import time
import datetime
from mss import mss
from mss.tools import to_png

Save_DIR = "screenshots"
INTERVAL_SECONDS = 10

def main() -> None:
    if not os.path.exists(Save_DIR):
        os.makedirs(Save_DIR)

    with mss() as sct:
        while True:
            # Capture the screen
            screenshot = sct.grab(sct.monitors[1])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(Save_DIR, f"screenshot_{timestamp}.png")

            # Save the screenshot
            to_png(screenshot.rgb, screenshot.size, output=filename)
            print(f"Screenshot saved: {filename}")

            # Wait for the specified interval
            time.sleep(INTERVAL_SECONDS)

while True:
    main()
    time.sleep(1)  # Prevent tight loop in case of errors