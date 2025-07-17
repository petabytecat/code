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

def parse_coords(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except Exception:
        raise argparse.ArgumentTypeError("Coordinates must be in the form '(x,y)'")

# Define Pokémon coordinates dictionary at top level
POKEMON_COORDS = {
    "slowpoke": (711, 471),
    "alomomola": (713, 511),
    "ghastly": (712, 473),
    "hatenna": (711, 482),
    "litten": (711, 469),
    "dratini": (711, 474),
    "chespin": (711, 477),
    "sprigatito": (711, 467),
    "scorbunny": (711, 505),
    "fuecoco": (711, 469),
    "quaxly": (711, 471),
    "nymble": (711, 467),
    "snivy": (711, 464),
    "rookidee": (711, 456),
    "larvesta": (711, 480),
    "cleffa": (711, 456)
}

parser = argparse.ArgumentParser(description='Pokemon Egg Hatcher')
parser.add_argument('--eggs', type=int, default=0, 
                    help='Number of eggs already in party (default: 0)')
parser.add_argument('--website', type=str, default='Bronze_Forever', 
                    help='Website to access and join the game (default: Bronze_Forever)')
parser.add_argument('--special_coords', type=parse_coords, default=None,
                    help="Coordinates to check for special ability/shiny (default: None). Format: '(x,y)'")
# Change type to str for Pokémon names
parser.add_argument('--pokemon', type=str, default=None,
                    help=f"Which Pokémon are you breeding? Options: {', '.join(POKEMON_COORDS.keys())}")

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "pokemon_state.txt")

state = {
    "box": 7,
    "row": 5,
    "column": 1
}
reset_timer = 100
current_box = 1

team = args.eggs
website = args.website

# Handle coordinate input logic
if args.special_coords and args.pokemon:
    print("Error: Use either --pokemon or --special_coords, not both")
    sys.exit(1)
elif args.pokemon:
    pokemon_lower = args.pokemon.lower()
    if pokemon_lower in POKEMON_COORDS:
        special_coords = POKEMON_COORDS[pokemon_lower]
    else:
        print(f"Error: Unknown Pokémon '{args.pokemon}'. Valid options: {', '.join(POKEMON_COORDS.keys())}")
        sys.exit(1)
elif args.special_coords:
    special_coords = args.special_coords
else:
    # Set default if neither provided
    print("Using default coordinates for Slowpoke")
    special_coords = (711, 464) 

print(f"Using coordinates: {special_coords}")

print(team, state)

time.sleep(3) 

def sleep(x):
    global reset_timer
    time.sleep(x)
    reset_timer += x

def check_hatch():
    global reset_timer, team
    r, g, b, y = getpixelcolor.pixel(244, 90)
    if r > 250 and g > 250 and b > 250: 
        pyautogui.leftClick()
        sleep(0.5)

        r, g, b, y = getpixelcolor.pixel(50, 677) 

        if r > 200 and 160 < g < 200 and 50 < b < 90:  

            pyautogui.moveTo(1327, 379)
            sleep(14)
            pyautogui.leftClick()
            sleep(2)
            pyautogui.leftClick()
            sleep(1)
            return True
    return False

def reset():
    global reset_timer
    before_reset = max(60 - reset_timer, 0) 
    sleep(before_reset)
    reset_timer = 0
    pyautogui.moveTo(29, 37)
    sleep(0.5)
    pyautogui.leftClick()
    pyautogui.moveTo(866, 629)
    sleep(0.5)
    pyautogui.leftClick()
    pyautogui.moveTo(745, 365)
    sleep(0.5)
    pyautogui.leftClick()
    sleep(1.5)

pyautogui.keyDown('shift')
while state["box"] <= 50:
    pyautogui.keyUp('shift')
    pyautogui.keyDown('shift')  

    while team < 5:
        pyautogui.keyUp('shift')
        pyautogui.keyDown('shift')  

        pyautogui.keyDown('a')
        time.sleep(1.9)
        pyautogui.keyUp('a')
        if check_hatch():
            reset()

        # START SOFT RESET

        # save
        pyautogui.moveTo(18, 344)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(55, 540)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(1342, 300)
        pyautogui.leftClick()
        sleep(2)
        pyautogui.leftClick()
        sleep(0.25)

        # clicks on dialogue 
        pyautogui.moveTo(780, 500)
        pyautogui.leftClick()
        time.sleep(0.5)
        pyautogui.moveTo(696, 420)
        pyautogui.leftClick()
        pyautogui.moveTo(231, 634)
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()

        pyautogui.keyUp('shift')
        
        r, g, b, y = getpixelcolor.pixel(1334, 300)
        if (r == 255 and g == 255 and b == 255) or (r == 105 and g == 105 and b == 105):
            # finishes dialogue to recieve egg
            pyautogui.moveTo(1334, 300)
            sleep(0.25)
            pyautogui.leftClick()
            sleep(0.5)
            pyautogui.leftClick()
            sleep(0.25)
            pyautogui.leftClick()
            sleep(0.25)
            pyautogui.leftClick()
            sleep(0.5)
            pyautogui.leftClick()
            
            # soft reset loop
            times = 0
            while True: 
                # swipes 
                pyautogui.keyDown('ctrl')
                pyautogui.press('left')
                pyautogui.press('left')
                sleep(0.5)
                pyautogui.keyUp('ctrl')
                
                pyautogui.moveTo(145, 192)
                pyautogui.leftClick()
                sleep(0.25)

                pyautogui.moveTo(548, 191)
                pyautogui.leftClick()
                sleep(0.25)

                # CLICKS ON 1
                pyautogui.moveTo(993, 250)
                pyautogui.leftClick()
                sleep(2.5)
                
                # checks if HA
                r, g, b, y = getpixelcolor.pixel(special_coords[0], special_coords[1])
                if 25 < r < 40 and 190 < g < 205 and 80 < b < 105:
                    print("Hidden Ability Detected...", r, g, b)
                    break

                elif 225 < r < 245 and 60 < g < 85 and 60 < b < 80:
                    print("Shiny Detected...", r, g, b)
                    break

                else:
                    print("Nothing...", r, g, b)

                    # clicks on new game
                    if website == "Roblox":
                        pyautogui.moveTo(117, 231)
                        pyautogui.leftClick()
                        time.sleep(0.25)
                        pyautogui.moveTo(823, 245)
                        pyautogui.leftClick()
                        time.sleep(0.25)
                        pyautogui.moveTo(1279, 345)
                        pyautogui.leftClick()
                        sleep(7.5)

                    else:
                        # clicks on tab
                        pyautogui.moveTo(145, 150)
                        pyautogui.leftClick()

                        if times >= 5:
                            print("Unable to join for 5 rounds...")
                            
                            pyautogui.moveTo(1046, 45) #clicks on Play
                            pyautogui.leftClick()
                            sleep(3)

                            pyautogui.moveTo(1601, 34) # closes tabs that are on top
                            pyautogui.leftClick()
                            sleep(1)

                            # refresh
                            pyautogui.moveTo(205, 23)
                            pyautogui.leftClick()
                            
                            while True:
                                pyautogui.scroll(100, x=587, y=356)
                                times = 0
                                sleep(0.5)

                                r, g, b, y = getpixelcolor.pixel(1255, 360)
                                print(r, g, b)
                                if 0 <= r < 20 and 175 < g < 190 and 85 < b < 105:
                                    break

                            pyautogui.moveTo(865, 362) # ignores roblox restarted
                            pyautogui.leftClick()
                            sleep(1)

                        pyautogui.moveTo(1320, 363)
                        time.sleep(0.25)
                        pyautogui.leftClick()
                        sleep(0.5)

                        pyautogui.moveTo(867, 397)
                        pyautogui.leftClick()
                        sleep(0.5)

                        pyautogui.moveTo(1140, 262)
                        pyautogui.leftClick()
                        sleep(0.1)

                        pyautogui.moveTo(1140, 262)
                        pyautogui.leftClick()

                        # swipes to game
                        pyautogui.keyDown('ctrl')
                        pyautogui.press('right')
                        pyautogui.press('right')
                        sleep(0.5)
                        pyautogui.keyUp('ctrl')

                    time.sleep(1)
                    r, g, b, y = getpixelcolor.pixel(54, 39)
                    if (25 < r < 45 and 190 < g < 205 and 60 < b < 80) or (0 < r < 10 and 95 < g < 130 and 0 < b < 20) or (195 < r < 210 and 195 < g < 210 and 195 < b < 210):
                        print("not full screen")
                        pyautogui.moveTo(54, 39)
                        sleep(1)
                        pyautogui.leftClick()
                    
                    #r, g, b, y = getpixelcolor.pixel(26, 12)
                    #if 

                    # checks if joined game
                    for i in range(20):
                        sleep(1)
                        r, g, b, y = getpixelcolor.pixel(340, 693)
                        if  r == 38 and g == 38 and b == 38:
                            pyautogui.moveTo(369, 693)
                            sleep(0.5)
                            pyautogui.leftClick()

                            sleep(0.5)
                            pyautogui.moveTo(696, 396)
                            pyautogui.leftClick()

                            sleep(1)
                            pyautogui.moveTo(605, 460)
                            pyautogui.leftClick()
                            while True:
                                pyautogui.moveTo(605, 460)
                                r, g, b, y = getpixelcolor.pixel(1354, 105)
                                if not (r == 255 and g == 255 and b == 255) and not (r == 105 and g == 105 and b == 105):
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                else:
                                    break

                            pyautogui.moveTo(1354, 300)
                            pyautogui.leftClick()
                            pyautogui.leftClick()

                            while True:
                                pyautogui.moveTo(1354, 300)
                                r, g, b, y = getpixelcolor.pixel(1354, 105)
                                if (r == 255 and g == 255 and b == 255) or (r == 105 and g == 105 and b == 105):
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                    pyautogui.leftClick()
                                else:
                                    break
                            break
                    else:
                        times += 1
                
            team += 1

            pyautogui.keyDown('ctrl')
            pyautogui.press('right')
            pyautogui.press('right')
            sleep(0.5)
            pyautogui.keyUp('ctrl')

            pyautogui.moveTo(786, 487)
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()
            pyautogui.leftClick()

            pyautogui.moveTo(18, 344)
            pyautogui.leftClick()
            sleep(1)

            pyautogui.moveTo(55, 540)
            pyautogui.leftClick()
            sleep(1)

            pyautogui.moveTo(1342, 300)
            pyautogui.leftClick()
            sleep(3)

            pyautogui.leftClick()
            sleep(0.5)

        pyautogui.keyDown('shift')
        # hoverboard setup
        pyautogui.keyDown('w')
        sleep(1)
        pyautogui.keyUp('w')
        pyautogui.press('r')

        # hoverboard
        pyautogui.keyDown('w')
        pyautogui.keyDown('d')
        for i in range(7):
            sleep(9.5)
            if check_hatch():
                pyautogui.keyUp('d')
                pyautogui.keyUp('w')
                pyautogui.keyDown('w')
                pyautogui.keyDown('d')

        pyautogui.keyUp('d')
        pyautogui.keyUp('w')
        pyautogui.keyUp('shift')
        # saves
        pyautogui.moveTo(18, 344)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(55, 540)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(1342, 300)
        pyautogui.leftClick()
        sleep(2)
        pyautogui.leftClick()
        sleep(0.25)

        pyautogui.keyDown('shift')

        while True:
            pyautogui.moveTo(868, 360)
            pyautogui.leftClick()

            pyautogui.keyUp('shift')
            pyautogui.keyDown('ctrl')
            pyautogui.press('left')
            pyautogui.press('left')
            sleep(0.5)
            pyautogui.keyUp('ctrl')

            # rejoin game CHANGE
            pyautogui.moveTo(145, 150)
            pyautogui.leftClick()

            sleep(0.25)
            pyautogui.moveTo(1320, 363)
            sleep(0.3)
            pyautogui.leftClick()
            sleep(0.5)

            pyautogui.moveTo(852, 388)
            pyautogui.leftClick()
            sleep(0.5)

            pyautogui.moveTo(1140, 262)
            pyautogui.leftClick()
            sleep(0.1)

            pyautogui.moveTo(1140, 262)
            pyautogui.leftClick()

            # swipes to game
            pyautogui.keyDown('ctrl')
            pyautogui.press('right')
            pyautogui.press('right')
            sleep(0.5)
            pyautogui.keyUp('ctrl')

            # full screen
            r, g, b, y = getpixelcolor.pixel(108, 78)
            if (35 < r < 45 and 190 < g < 205 and 60 < b < 70) or (0 < r < 10 and 95 < g < 130 and 0 < b < 20) or (195 < r < 210 and 195 < g < 210 and 195 < b < 210):
                print("not full screen")
                sleep(3)
                pyautogui.moveTo(108, 78)
            pyautogui.leftClick()

            # checks if joined game
            breakout = False
            for i in range(20):
                sleep(1)
                r, g, b, y = getpixelcolor.pixel(340, 693)
                if 35 < r < 45 and 35 < g < 45 and 35 < b < 45:
                    sleep(7)
                    pyautogui.moveTo(369, 693)
                    pyautogui.leftClick()
                    sleep(1.5)

                    pyautogui.moveTo(696, 396)
                    pyautogui.leftClick()
                    sleep(3)

                    pyautogui.moveTo(770, 491)
                    pyautogui.leftClick()
                    sleep(2)
                    
                    reset()
                    breakout = True
                    break
            if breakout == True:
                break

        pyautogui.keyDown('shift')

    
    # T3: egg to PC
    pyautogui.keyDown('w')
    sleep(0.5)
    pyautogui.keyUp('w')
    sleep(3)

    pyautogui.keyDown('d')
    sleep(0.8)
    pyautogui.keyUp('d')
    pyautogui.keyDown('w')
    sleep(1)
    pyautogui.keyUp('w')
    if check_hatch():
        reset()
        continue

    sleep(1)
    pyautogui.moveTo(886, 380)
    sleep(0.5)
    pyautogui.leftClick()
    sleep(4)
    r, g, b, y = getpixelcolor.pixel(439, 251)

    if r != 217 or g != 87 or b != 84:
                
        pyautogui.keyDown('s')
        sleep(0.5)
        pyautogui.keyUp('s')

        pyautogui.keyDown('w')
        sleep(1)
        pyautogui.keyUp('w')

        pyautogui.keyDown('d')
        sleep(0.5)
        pyautogui.keyUp('d')
        pyautogui.keyDown('w')
        sleep(0.8)
        pyautogui.keyUp('w')
        sleep(0.5)
        if check_hatch():
            reset()

            pyautogui.keyDown('s')
            sleep(0.5)
            pyautogui.keyUp('s')
            sleep(3)

            continue

    pyautogui.keyUp('shift')
    while team > 0:
        pyautogui.moveTo(1321, 391)
        while current_box < state["box"]:
            
            event = CGEventCreateScrollWheelEvent(
                None,
                kCGScrollEventUnitPixel,
                1,
                -210
            )
            CGEventPost(kCGHIDEventTap, event)
            sleep(0.5)
            current_box += 1
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(446, 326)
        pyautogui.mouseDown()
        row = max(1, min(state["row"], 5))
        column = max(1, min(state["column"], 6))
        sleep(1)
        x0, y0 = 629, 227
        x1, y1 = 1100, 590
        cell_width = (x1 - x0) / (6 - 1)
        cell_height = (y1 - y0) / (5 - 1)
        x = x0 + (column - 1) * cell_width
        y = y0 + (row - 1) * cell_height
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.mouseUp()
        sleep(1)
        state["column"] += 1
        if state["column"] > 6:
            state["column"] = 1
            state["row"] += 1
            if state["row"] > 5:
                state["row"] = 1
                state["box"] += 1
        print(f"Next storage position: Box {state['box']}, Row {state['row']}, Column {state['column']}")
        team -= 1
        print(f"Team after transfer: {team}")
    pyautogui.keyDown('shift')
    pyautogui.moveTo(1146, 95)
    sleep(1.5)
    pyautogui.leftClick()
    sleep(2)
    pyautogui.keyDown('s')
    sleep(0.3)
    pyautogui.keyUp('s')
    pyautogui.keyDown('a')
    pyautogui.keyDown('s')
    sleep(1.4)
    pyautogui.keyUp('a')
    pyautogui.keyUp('s')
    
    if check_hatch():

        reset()

        pyautogui.keyDown('s')
        sleep(0.5)
        pyautogui.keyUp('s')
    sleep(3)
    current_box = 1