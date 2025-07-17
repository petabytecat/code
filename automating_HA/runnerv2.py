import time
import os
import pyautogui
import sys
from datetime import datetime
from PIL import ImageGrab
from Quartz.CoreGraphics import (
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    kCGHIDEventTap,
    kCGScrollEventUnitPixel
)
import getpixelcolor
import argparse
pyautogui.PAUSE = 0.1
parser = argparse.ArgumentParser(description='Pokemon Egg Hatcher')
parser.add_argument('--hatched', type=int, default=0, 
                    help='Number of already hatched Pokemon in party (default: 0)')
parser.add_argument('--eggs', type=int, default=0, 
                    help='Number of eggs already in party (default: 0)')
args = parser.parse_args()
script_dir = os.path.dirname(os.path.abspath(__file__))
base_screenshot_dir = os.path.join(script_dir, "Screenshots")
iv_dir = os.path.join(base_screenshot_dir, "IVs")
summary_dir = os.path.join(base_screenshot_dir, "Summaries")
os.makedirs(iv_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)
empty = {"box": 3, "row": 1, "column": 1}
max_box = 0
max_row = 0
max_col = 0
found_screenshots = False
if os.path.exists(summary_dir):
    for filename in os.listdir(summary_dir):
        if filename.endswith("_Summary.png"):
            try:
                parts = filename.split('_')
                if len(parts) >= 4:
                    box = int(parts[0])
                    row = int(parts[1])
                    col = int(parts[2])
                    found_screenshots = True
                    if (box > max_box or 
                        (box == max_box and row > max_row) or 
                        (box == max_box and row == max_row and col > max_col)):
                        max_box = box
                        max_row = row
                        max_col = col
            except (ValueError, IndexError):
                continue
if found_screenshots:
    if max_col < 6:
        empty["box"] = max_box
        empty["row"] = max_row
        empty["column"] = max_col + 1
    else:
        if max_row < 5:
            empty["box"] = max_box
            empty["row"] = max_row + 1
            empty["column"] = 1
        else:
            empty["box"] = max_box + 1
            empty["row"] = 1
            empty["column"] = 1

print(empty)
team = [1] * args.hatched + [0] * args.eggs
current_box = 1
reset_timer = 60
time.sleep(3) # 5 seconds to swipe tab
def check_hatch():
    global reset_timer, team
    r, g, b, y = getpixelcolor.pixel(244, 90)
    if r > 250 and g > 250 and b > 250:  # Bright color = textbox visible
        pyautogui.leftClick()
        time.sleep(0.5)
        reset_timer += 0.5

        r, g, b, y = getpixelcolor.pixel(50, 677) 

        if r > 200 and 160 < g < 200 and 50 < b < 90:  # Orange color range

            pyautogui.moveTo(1327, 379)
            time.sleep(14)
            pyautogui.leftClick()
            time.sleep(2)
            pyautogui.leftClick()
            time.sleep(1)
            reset_timer += 17
            for i in range(len(team)):
                if team[i] == 0:
                    team[i] = 1
                    break
            return True
    return False
def run():
    global reset_timer
    pyautogui.keyDown('a')
    time.sleep(1.3)
    reset_timer += 1.3
    pyautogui.keyUp('a')
    pyautogui.keyDown('d')
    time.sleep(1.3)
    reset_timer += 1.3
    pyautogui.keyUp('d')
def reset():
    global reset_timer
    before_reset = max(60 - reset_timer, 0) 
    time.sleep(before_reset)
    reset_timer = 0
    pyautogui.moveTo(29, 37)
    time.sleep(0.5)
    pyautogui.leftClick()
    pyautogui.moveTo(866, 629)
    time.sleep(0.5)
    pyautogui.leftClick()
    pyautogui.moveTo(745, 365)
    time.sleep(0.5)
    pyautogui.leftClick()
    time.sleep(1.5)
    reset_timer += 3
pyautogui.keyDown('shift')  
while empty["box"] <= 50:
    pyautogui.keyUp('shift')
    pyautogui.keyDown('shift')  

    while True:
        hatch = False
        if len(team) >= 5:
            pyautogui.keyDown('w')
            time.sleep(0.5)
            pyautogui.keyUp('w')
            reset_timer += 0.5
            time.sleep(3)
            reset_timer += 3
            break

        # get egg
        pyautogui.keyDown('a')
        time.sleep(1.9)
        reset_timer += 1.9
        pyautogui.keyUp('a')
        if check_hatch():
            before_reset = max(60 - reset_timer, 0)
            time.sleep(before_reset)
            reset()
            pyautogui.keyDown('w')
            time.sleep(0.5)
            pyautogui.keyUp('w')
            reset_timer += 0.5
            time.sleep(3)
            reset_timer += 3
            break
        pyautogui.moveTo(747, 486)
        pyautogui.leftClick()
        pyautogui.moveTo(231, 634)
        time.sleep(0.5)
        pyautogui.leftClick()
        time.sleep(0.5)
        pyautogui.leftClick()
        time.sleep(0.5)
        pyautogui.leftClick()
        time.sleep(0.5)
        pyautogui.leftClick()
        reset_timer += 2
        r, g, b, y = getpixelcolor.pixel(1334, 300)
        if (r == 255 and g == 255 and b == 255) or (r == 105 and g == 105 and b == 105):
            pyautogui.moveTo(1334, 300)
            time.sleep(1)
            pyautogui.leftClick()
            time.sleep(1)
            pyautogui.leftClick()
            time.sleep(1)
            pyautogui.leftClick()
            time.sleep(1)
            pyautogui.leftClick()
            reset_timer += 4
            team.append(0)
        
        # go back
        time.sleep(1)
        reset_timer += 1
        pyautogui.keyDown('d')
        time.sleep(1.6)
        reset_timer += 1.6
        pyautogui.keyUp('d')
        pyautogui.keyDown('w')
        time.sleep(1)
        pyautogui.keyUp('w')
        if check_hatch():
            reset()
            pyautogui.keyDown('w')
            time.sleep(0.5)
            pyautogui.keyUp('w')
            reset_timer += 0.5
            time.sleep(3)
            reset_timer += 3
            break
        time.sleep(3)
        reset_timer += 4
        if len(team) >= 5:
            break
        pyautogui.keyDown('w')
        time.sleep(0.25)
        pyautogui.keyUp('w')
        reset_timer += 0.25
        pyautogui.keyDown('d')
        time.sleep(0.6)
        pyautogui.keyUp('d')
        reset_timer += 0.6

        for i in range(4):
            for i in range(2):
                run()
            if (check_hatch() or team[0] == 1) and reset_timer > 60:
                reset()
                hatch = True
                break
        if hatch == True:
            break

        # go outside
        pyautogui.keyDown('a')
        time.sleep(0.4)
        pyautogui.keyUp('a')
        reset_timer += 0.4
        pyautogui.keyDown('s')
        time.sleep(0.5)
        pyautogui.keyUp('s')
        reset_timer += 0.5
        if check_hatch():
            reset()
            pyautogui.keyDown('s')
            time.sleep(0.5)
            pyautogui.keyUp('s')
            reset_timer += 0.5
            time.sleep(3)
            break
        time.sleep(3)
        if reset_timer > 360:
            reset()
            pyautogui.keyDown('w')
            time.sleep(0.5)
            pyautogui.keyUp('w')
            reset_timer += 0.5
            time.sleep(3)
            pyautogui.keyDown('s')
            time.sleep(1)
            pyautogui.keyUp('s')
            reset_timer += 1
            time.sleep(3)
    
    if team and team[0] != 1:

        # indoor hatch run
        pyautogui.keyDown('w')
        time.sleep(0.25)
        pyautogui.keyUp('w')
        reset_timer += 0.25
        pyautogui.keyDown('d')
        time.sleep(0.6)
        pyautogui.keyUp('d')
        reset_timer += 0.6

        while True:
            if (team and team[0] == 1) and reset_timer > 60:
                break
            for i in range(3):
                run()
            check_hatch()
        reset()
    pyautogui.keyDown('d')
    time.sleep(0.5)
    pyautogui.keyUp('d')
    reset_timer += 0.5
    pyautogui.keyDown('w')
    time.sleep(0.8)
    pyautogui.keyUp('w')
    reset_timer += 0.8
    time.sleep(1)
    if check_hatch():
        reset()

        pyautogui.keyDown('s')
        time.sleep(0.5)
        pyautogui.keyUp('s')
        reset_timer += 0.5
        time.sleep(3)

        continue
    pyautogui.moveTo(886, 380)
    time.sleep(0.5)
    pyautogui.leftClick()
    time.sleep(4)
    reset_timer += 4.5
    r, g, b, y = getpixelcolor.pixel(439, 251)

    if r != 217 or g != 87 or b != 84:
                
        pyautogui.keyDown('s')
        time.sleep(0.5)
        pyautogui.keyUp('s')

        pyautogui.keyDown('w')
        time.sleep(1)
        pyautogui.keyUp('w')

        pyautogui.keyDown('d')
        time.sleep(0.5)
        pyautogui.keyUp('d')
        reset_timer += 0.5
        pyautogui.keyDown('w')
        time.sleep(0.8)
        pyautogui.keyUp('w')
        reset_timer += 0.8
        time.sleep(1)
        if check_hatch():
            reset()

            pyautogui.keyDown('s')
            time.sleep(0.5)
            pyautogui.keyUp('s')
            reset_timer += 0.5
            time.sleep(3)

            continue

    pyautogui.keyUp('shift')
    while team and team[0] == 1:
        pyautogui.moveTo(1321, 391)
        while current_box < empty["box"]:
            
            event = CGEventCreateScrollWheelEvent(
                None,
                kCGScrollEventUnitPixel,
                1,
                -210
            )
            CGEventPost(kCGHIDEventTap, event)
            time.sleep(0.5)
            reset_timer += 0.5
            current_box += 1
        pyautogui.leftClick()
        filename_base = f"{empty['box']}_{empty['row']}_{empty['column']}"
        summary_path = os.path.join(summary_dir, f"{filename_base}_Summary.png")
        pyautogui.moveTo(446, 326)
        pyautogui.leftClick()
        time.sleep(1)
        pyautogui.moveTo(535, 301)
        pyautogui.leftClick()
        time.sleep(2)
        reset_timer += 3
        screenshot = ImageGrab.grab()
        screenshot.save(summary_path)
        pyautogui.moveTo(494, 380)
        pyautogui.leftClick()
        time.sleep(1)
        reset_timer += 1
        pyautogui.moveTo(446, 326)
        pyautogui.mouseDown()
        row = max(1, min(empty["row"], 5))
        column = max(1, min(empty["column"], 6))
        time.sleep(1)
        x0, y0 = 629, 227
        x1, y1 = 1100, 590
        cell_width = (x1 - x0) / (6 - 1)
        cell_height = (y1 - y0) / (5 - 1)
        x = x0 + (column - 1) * cell_width
        y = y0 + (row - 1) * cell_height
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.mouseUp()
        time.sleep(1)
        reset_timer += 2.5
        empty["column"] += 1
        if empty["column"] > 6:
            empty["column"] = 1
            empty["row"] += 1
            if empty["row"] > 5:
                empty["row"] = 1
                empty["box"] += 1
        print(f"Next storage position: Box {empty['box']}, Row {empty['row']}, Column {empty['column']}")
        team.pop(0)
        print(f"Team after transfer: {team}")
    pyautogui.keyDown('shift')
    pyautogui.moveTo(1146, 95)
    time.sleep(1.5)
    pyautogui.leftClick()
    reset_timer += 1.5
    time.sleep(2)
    reset_timer += 2
    pyautogui.keyDown('s')
    time.sleep(0.3)
    pyautogui.keyUp('s')
    reset_timer += 0.3
    pyautogui.keyDown('a')
    pyautogui.keyDown('s')
    time.sleep(1.4)
    pyautogui.keyUp('a')
    pyautogui.keyUp('s')
    reset_timer += 1.4
    
    if check_hatch():
        reset()

        pyautogui.keyDown('s')
        time.sleep(0.5)
        pyautogui.keyUp('s')
        reset_timer += 0.5
    time.sleep(3)
    reset_timer += 3

    if len(team) == 0:
        reset()

        pyautogui.keyDown('w')
        time.sleep(0.5)
        pyautogui.keyUp('w')

        pyautogui.keyDown('s')
        time.sleep(1)
        pyautogui.keyUp('s')

        time.sleep(3)

        team = [1, 1, 1, 1, 1]

