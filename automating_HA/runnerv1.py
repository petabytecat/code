"""
I have decided not to use this as I found out macro is bannable and regular players snitch with video evidence

Thus, I have decided to use a macro that runs inside the pokestore. Although it is less efficient, I don't want to lose progress on the alt

This way it will be virtually unbannable
"""

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
print(f"Starting with {args.hatched} hatched and {args.eggs} eggs in party")


current_box = 1
instore = False
reset_timer = 60

# STEPS TO SET UP
# 1. Flame Body / Steam Engine member in first slot 
# 2. Place the two pokemons in day care and 
# 3. IM USING 1720 X 720 (HiDPI) monitor (CHANGE COORDINATES ACCORDINGLY)
# 4. FIll out the following
# 5. Walk around, save, leave the game, and join back

# Start automator...
time.sleep(3) # 5 seconds to swipe tab

def check_hatch():
    global reset_timer, team
    r, g, b, y = getpixelcolor.pixel(244, 90)
    if r > 250 and g > 250 and b > 250:  # Bright color = textbox visible
        pyautogui.leftClick()
        time.sleep(1)
        reset_timer += 1

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


pyautogui.keyDown('shift')  
while empty["box"] <= 50:
    pyautogui.keyUp('shift')  
    pyautogui.keyDown('shift')  
    # moves to egg person
    pyautogui.keyDown('a')  
    
    time.sleep(2.1)
    reset_timer += 2.1
    pyautogui.keyUp('a')

    while len(team) < 5:
        # clicks on egg person + dialogue
        check_hatch()
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
        
        # check if egg
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
            time.sleep(1)
            reset_timer += 5
            team.append(0)

        for i in range(2):
            pyautogui.keyDown('d')
            time.sleep(4)
            pyautogui.keyUp('d')
            reset_timer += 3.1

            check_hatch()
            pyautogui.keyDown('a')
            time.sleep(4.005)
            pyautogui.keyUp('a')
            reset_timer += 3.1

            if (team and team[0] == 1) or reset_timer > 360:
                break
        
        if (team and team[0] == 1) or reset_timer > 360:
            break
            
    # hatches
    while True:
        if (team and team[0] == 1) and reset_timer > 60:
            break
        # if hatch is true abort this loop and go to PC

        #check_hatch()
        pyautogui.keyDown('d')
        time.sleep(4)
        reset_timer += 4
        pyautogui.keyUp('d')

        #check_hatch()
        pyautogui.keyDown('a')
        time.sleep(4)
        reset_timer += 4
        pyautogui.keyUp('a')

        check_hatch()

    # reset
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
    
    # goes into store
    instore = True
    pyautogui.keyDown('w')  
    time.sleep(1)
    pyautogui.keyUp('w')
    time.sleep(4)
    reset_timer += 5

    # goes to PC
    pyautogui.keyDown('d')  
    time.sleep(0.5)
    pyautogui.keyUp('d')
    reset_timer += 0.5

    pyautogui.keyDown('w')  
    time.sleep(0.8)
    pyautogui.keyUp('w')
    
    reset_timer += 0.8

    # if hatches inside the store
    if check_hatch(): 
        continue

    # clicks PC
    pyautogui.keyUp('shift')  
    pyautogui.moveTo(886, 380)
    time.sleep(0.5)
    pyautogui.leftClick()
    time.sleep(4)
    reset_timer += 4.5

    # PC box is 590 pixels
    # each scroll unit pyautogui is 139 pixels in PC
    # each scroll unit (by pixel) Quartz framework is 3 pixels
    # WILL USE: -210 Quartz framework. Closest through trial and error

    # scrolls down PC to target box
    while current_box < empty["box"]:
        pyautogui.moveTo(1321, 391)

        event = CGEventCreateScrollWheelEvent(
            None,
            kCGScrollEventUnitPixel,
            1,
            -210
        )

        CGEventPost(kCGHIDEventTap, event)
        time.sleep(0.5)
        reset_timer += 0.5

        pyautogui.leftClick()

        current_box += 1
    
    # Create filename base (box/row/column)
    filename_base = f"{empty['box']}_{empty['row']}_{empty['column']}"

    # clicks on pokemon
    pyautogui.moveTo(446, 326)
    pyautogui.leftClick()
    time.sleep(1)  
    reset_timer += 1

    # clicks on summary
    pyautogui.moveTo(535, 301)
    pyautogui.leftClick()
    time.sleep(2)
    reset_timer += 2

    # Take and save summary screenshot
    summary_path = os.path.join(summary_dir, f"{filename_base}_Summary.png")
    screenshot = ImageGrab.grab()
    screenshot.save(summary_path)

    """
    # clicks on effort values
    pyautogui.moveTo(870, 389)
    pyautogui.leftClick()
    time.sleep(1)  
    reset_timer += 1

    # Take and save IV screenshot
    iv_path = os.path.join(iv_dir, f"{filename_base}_EV.png")
    screenshot = ImageGrab.grab()
    screenshot.save(iv_path)

    # closes effort values
    pyautogui.moveTo(1154, 202)
    time.sleep(1)      
    pyautogui.leftClick()
    reset_timer += 1
    """    
    # closes summary
    pyautogui.moveTo(494, 380)
    time.sleep(1.5)  
    pyautogui.leftClick()
    reset_timer += 1.5

    time.sleep(1)
    reset_timer += 1

    pyautogui.moveTo(446, 326)

    pyautogui.mouseDown()
    row = max(1, min(empty["row"], 5))
    column = max(1, min(empty["column"], 6))
    time.sleep(1)  
    reset_timer += 1

    x0, y0 = 629, 227 
    x1, y1 = 1100, 590 

    cell_width = (x1 - x0) / (6 - 1) 
    cell_height = (y1 - y0) / (5 - 1)

    x = x0 + (column - 1) * cell_width
    y = y0 + (row - 1) * cell_height

    pyautogui.moveTo(x, y, duration=0.5)
    pyautogui.mouseUp()
    time.sleep(1)  

    # close PC
    pyautogui.moveTo(1146, 95)
    time.sleep(1.5)  
    pyautogui.leftClick()
    reset_timer += 1.5
    
    time.sleep(1.5)
    reset_timer += 1.5
    # Leaves
    pyautogui.keyDown('s')  
    pyautogui.keyDown('shift')  
    time.sleep(0.1)
    pyautogui.keyUp('s')
    pyautogui.keyUp('shift') 
    reset_timer += 0.1

    pyautogui.keyDown('a')  
    pyautogui.keyDown('s')  
    pyautogui.keyDown('shift')  
    time.sleep(1.4)
    pyautogui.keyUp('a')
    pyautogui.keyUp('s')
    pyautogui.keyUp('shift')  
    reset_timer += 1.4

    time.sleep(4)
    reset_timer += 4

    instore = False

    # PC aim next
    empty["column"] += 1
    if empty["column"] > 6: 
        empty["column"] = 1
        empty["row"] += 1
        
        if empty["row"] > 5: 
            empty["row"] = 1
            empty["box"] += 1  
        
    print(team)

    if team:
        team.pop(0)
    pyautogui.keyDown('shift')  
