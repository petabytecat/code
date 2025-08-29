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


# Coordinates of where the star is in egg tracker website (to detect HA/Shiny)
# 0: x coordinate of 1 star pokemon
# 1: x coordinate of 2 star pokemon
# 2: y coordinate 
POKEMON_COORDS = {
    "slowpoke": (711, 699, 509), # 471
    "alomomola": (711, 699, 511),
    "ghastly": (711, 699, 473),
    "hatenna": (711, 699, 522),
    "litten": (711, 699, 469),
    "dratini": (711, 699, 474),
    "chespin": (711, 699, 477),
    "sprigatito": (711, 699, 467),
    "scorbunny": (711, 699, 505),
    "fuecoco": (711, 699, 469),
    "quaxly": (711, 699, 471),
    "nymble": (711, 699, 467),
    "snivy": (711, 699, 468),
    "rookidee": (711, 699, 456),
    "larvesta": (711, 699, 480),
    "cleffa": (711, 699, 456),
    "wooper": (711, 699, 459),
    "vulpix": (711, 699, 476), # -40
    "foongus": (711, 699, 460),
    "swinub": (711, 699, 446),
    "starly": (711, 699, 459),
    "varoom": (711, 699, 470),
    "cubone": (711, 699, 461),
    "larvitar": (711, 699, 477),
    "rotom": (711, 699, 519),
    "cufant": (711, 699, 478)
}

# If breeding without T3, the code will need to dump the pokemon into PC
# Change this to your first empty box
state = {
    "box": 10,
    "row": 1,
    "column": 3
}


def parse_coords(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except Exception:
        raise argparse.ArgumentTypeError("Coordinates must be in the form '(x,y)'")
parser = argparse.ArgumentParser(description='Pokemon Egg Hatcher')
parser.add_argument('--eggs', type=int, default=0, 
                    help='Number of eggs already in party (default: 0)')
parser.add_argument('--website', type=str, default='Roblox', 
                    help='Website to access and join the game (Roblox, Bronze_Forever)')
parser.add_argument('--special_coords', type=parse_coords, default=None,
                    help="Coordinates to check for special ability/shiny (default: None). Format: '(x,y)'")
parser.add_argument('--pokemon', type=str, default=None,
                    help=f"Which Pokémon are you breeding? Options: {', '.join(POKEMON_COORDS.keys())}")
parser.add_argument('--subscription', type=str, default=None,
                    help=f"Your subscription (None, T1, T2, T3)")
args = parser.parse_args()
reset_timer = 100
current_box = 1
team = args.eggs
website = args.website
subscription = args.subscription if args.subscription in ["T1", "T2", "T3"] else None
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
    print("Using default coordinates for Slowpoke")
    special_coords = (711, 699, 464) 
print(f"Using coordinates: {special_coords}")
print(team, state)
time.sleep(3) 
def sleep(x):
    global reset_timer
    time.sleep(x)
    reset_timer += x

def safe_pixel(x, y, retries=3, delay=0.05):
    for attempt in range(retries):
        try:
            result = getpixelcolor.pixel(x, y)
            if result and all(isinstance(i, int) for i in result):
                return result
        except Exception:
            pass
        time.sleep(delay)
    # If still failing, return fallback value
    print(f"[WARN] Pixel read failed at ({x}, {y}), returning fallback.")
    return (0, 0, 0, 255)


# checks if an egg is hatching
def check_hatch():
    global reset_timer, team
    r, g, b, y = safe_pixel(244, 90) # detects if there is a white textbox
    if r > 250 and g > 250 and b > 250: 
        pyautogui.leftClick()
        sleep(0.5)

        r, g, b, y = safe_pixel(50, 677) # detects if background is orange (double check if its hatching)

        if r > 200 and 160 < g < 200 and 50 < b < 90:  

            pyautogui.moveTo(1327, 379) # give nickname to ___, clicks "No"
            sleep(14)
            pyautogui.leftClick() 
            sleep(2)
            pyautogui.leftClick() 
            sleep(1)
            return True
    return False

# resets the character
def reset():
    global reset_timer
    before_reset = max(60 - reset_timer, 0) 
    sleep(before_reset)
    reset_timer = 0
    # pyautogui.moveTo(29, 37) # clicks on roblox symbol on top left 
    pyautogui.press('esc')
    sleep(0.5)
    pyautogui.moveTo(866, 629) # clicks "Respawn"
    sleep(0.5)
    pyautogui.leftClick()
    pyautogui.moveTo(745, 365) # clicks "Reset"
    sleep(0.5)
    pyautogui.leftClick()
    sleep(1.5)

# rejoin game
def rejoin(website):
    # clicks on new game
    if website == "Roblox": 
        pyautogui.moveTo(117, 231) # Click on the tab with the Roblox website
        pyautogui.leftClick()
        time.sleep(0.25) 
        pyautogui.moveTo(823, 245) # clicks on join game
        pyautogui.leftClick()
        time.sleep(0.25)
        pyautogui.moveTo(1279, 345) # closes the pop up "X" on top left of "roblox is now loading"/"download roblox"
        pyautogui.leftClick()
        sleep(7.5)

    else: # THIS PART DOESNT WORK ANYMORE. U CAN SKIP IF U USE ROBLOX WEBSITE
        pyautogui.moveTo(145, 150) # clicks on brick bronze forever tab
        pyautogui.leftClick()

        if times >= 5:
            print("Unable to join for 5 rounds...")
            
            pyautogui.moveTo(1046, 45) #clicks on Play
            pyautogui.leftClick()
            sleep(3)

            pyautogui.moveTo(1601, 34) # closes tabs that are on top
            pyautogui.leftClick()
            sleep(1)

            pyautogui.moveTo(205, 23) # clicks refresh tab
            pyautogui.leftClick()
            
            while True:
                pyautogui.scroll(100, x=587, y=356) # scroll in an appropriate place
                times = 0
                sleep(0.5)

                r, g, b, y = safe_pixel(1255, 360) # checks if the button is green in the correct spot
                print(r, g, b)
                if 0 <= r < 20 and 175 < g < 190 and 85 < b < 105:
                    break

            pyautogui.moveTo(858, 348) # roblox unexpectedly quit. click ignore
            pyautogui.leftClick()
            sleep(0.25)

            pyautogui.moveTo(865, 362) # ignores roblox restarted
            pyautogui.leftClick()
            sleep(1)

        pyautogui.moveTo(1320, 363) # clicks on play
        time.sleep(0.25)
        pyautogui.leftClick()
        sleep(0.5)

        pyautogui.moveTo(867, 397) # clicks "open roblox". "___ wants to open roblox"
        pyautogui.leftClick()
        sleep(0.5)

        pyautogui.moveTo(1140, 262) # clicks "X" on "launching game" top right
        pyautogui.leftClick()
        sleep(0.1)

        # swipes to game
        pyautogui.keyDown('ctrl')
        pyautogui.press('right')
        pyautogui.press('right')
        sleep(0.5)
        pyautogui.keyUp('ctrl')

    time.sleep(1)

    # full screen
    time.sleep(1)
    r, g, b, y = safe_pixel(54, 39) # checks if it is full screen (to do this, what i did was I checked if the apple symbol on top left is showing and whether it is white or black)
    if (25 < r < 45 and 190 < g < 205 and 60 < b < 80) or (0 < r < 10 and 95 < g < 130 and 0 < b < 20) or (195 < r < 210 and 195 < g < 210 and 195 < b < 210):
        print("not full screen")
        pyautogui.moveTo(54, 39) #clicks on full screen button (before every game i make it so that roblox that isnt full screened)
        sleep(1)
        pyautogui.leftClick()

pyautogui.keyDown('shift')
while state["box"] <= 50:
    pyautogui.keyUp('shift')
    pyautogui.keyDown('shift')  

    while (subscription is None and team < 5) or (subscription in ["T1", "T2", "T3"]):  # egg to PC
        pyautogui.keyUp('shift')
        pyautogui.keyDown('shift')  

        # start from the exit of lagoona lake healer, moves to NPC
        pyautogui.keyDown('a')
        time.sleep(1.9)
        pyautogui.keyUp('a')
        if check_hatch():
            reset()

        # save game
        pyautogui.moveTo(18, 344) # clicks menu
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(55, 540) # clicks save
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(1342, 300) # clicks yes
        pyautogui.leftClick()
        sleep(2)
        pyautogui.leftClick()
        sleep(0.25)

        # clicks on dialogue 
        pyautogui.moveTo(780, 500) # Clicks No: "Favorite item, do you want to add ____ to favorites?". at the start of the game
        pyautogui.leftClick()
        time.sleep(0.5)
        pyautogui.moveTo(696, 420) # clicks on NPC
        pyautogui.leftClick()
        pyautogui.moveTo(231, 634) # clicks on yes
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.leftClick()
        pyautogui.keyUp('shift')
        
        r, g, b, y = safe_pixel(1334, 300) # checks if NPC has egg "yes" button. checks if it is white
        if (r == 255 and g == 255 and b == 255) or (r == 105 and g == 105 and b == 105):
            
            # finishes dialogue to recieve egg
            pyautogui.moveTo(1334, 300) # clicks "yes"
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
            
            # =========== SOFT RESET LOOP ===========
            times = 0
            while True: 
                # swipes to google / whatever browser you are using (im using Arc)
                pyautogui.keyDown('ctrl')
                pyautogui.press('left')
                pyautogui.press('left')
                sleep(0.5)
                pyautogui.keyUp('ctrl')
                
                pyautogui.moveTo(145, 192) # clicks on egg tracker tab 
                pyautogui.leftClick()
                sleep(0.25)
                pyautogui.leftClick()
                sleep(0.1)

                pyautogui.moveTo(548, 191) 
                pyautogui.leftClick()
                sleep(0.25)

                pyautogui.moveTo(993, 250) 
                pyautogui.leftClick()
                sleep(2.5)
                
                # checks if HA
                r1, g1, b1, y1 = safe_pixel(special_coords[0], special_coords[2]) # 1 STAR coords
                r2, g2, b2, y2 = safe_pixel(special_coords[1], special_coords[2]) # 2 STAR coords
                if ((25 < r1 < 40 and 190 < g1 < 205 and 80 < b1 < 105) or
                    (25 < r2 < 40 and 190 < g2 < 205 and 80 < b2 < 105)):
                    print("Hidden Ability Detected...", r1, g1, b1, r2, g2, b2)
                    break

                elif ((225 < r1 < 245 and 60 < g1 < 85 and 60 < b1 < 80) or
                    (225 < r2 < 245 and 60 < g2 < 85 and 60 < b2 < 80)):
                    print("Shiny Detected...", r1, g1, b1, r2, g2, b2)
                    break

                else:
                    print("Nothing...", r1, g1, b1, r2, g2, b2)
                    rejoin(website)
                    
                    # checks if joined game
                    for i in range(20):
                        sleep(1)
                        r, g, b, y = safe_pixel(340, 693) # checks if "skip intro" button is grey
                        if  r == 38 and g == 38 and b == 38:
                            pyautogui.moveTo(369, 693) # clicks skip intro
                            sleep(0.5)
                            pyautogui.leftClick()

                            sleep(0.5)
                            pyautogui.moveTo(696, 396) # clicks continue
                            pyautogui.leftClick()

                            sleep(1)
                            pyautogui.moveTo(644, 453) # clicks on NPC
                            pyautogui.leftClick()
                            while True:
                                pyautogui.moveTo(644, 453) # clicks on NPC
                                r, g, b, y = safe_pixel(1354, 105) # checks if the speech bubble/npc text is white
                                if not (r == 255 and g == 255 and b == 255) and not (r == 105 and g == 105 and b == 105):
                                    pyautogui.leftClick()
                                else:
                                    break

                            pyautogui.moveTo(1354, 300) # clicks yes
                            pyautogui.leftClick()

                            while True:
                                pyautogui.moveTo(1354, 300) # keep on clicking yes
                                r, g, b, y = safe_pixel(1354, 105) # checks if speech bubble is still there
                                if (r == 255 and g == 255 and b == 255) or (r == 105 and g == 105 and b == 105):
                                    pyautogui.leftClick()
                                else:
                                    break
                            break
                    else: # it quit unexpectedly
                        times += 1
                        sleep(0.5)
                        
                        # clicks on ignore 
                        pyautogui.moveTo(858, 348) # "roblox quit unexpectedly" click on "ignore"
                        pyautogui.leftClick()
                        sleep(0.25)
            # =======================================
                
            team += 1

            pyautogui.keyDown('ctrl')
            pyautogui.press('right')
            pyautogui.press('right')
            sleep(0.5)
            pyautogui.keyUp('ctrl')

            pyautogui.moveTo(786, 487) # clicks yes NPC speech bubble
            for i in range(10):
                pyautogui.leftClick()
                sleep(0.1)
            
            pyautogui.moveTo(18, 344) # clicks menu
            pyautogui.leftClick()
            sleep(1)

            pyautogui.moveTo(55, 540) # clicks save
            pyautogui.leftClick()
            sleep(1)

            pyautogui.moveTo(1342, 300) # clicks yes
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
        pyautogui.moveTo(18, 344) # clicks menu
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(55, 540) # clicks save
        pyautogui.leftClick()
        sleep(0.5)
        pyautogui.moveTo(1342, 300) # clicks yes
        pyautogui.leftClick()
        sleep(2)
        pyautogui.leftClick()
        sleep(0.25)

        pyautogui.keyDown('shift')

        while True:
            pyautogui.moveTo(868, 360) # clicks on ignore when roblox has quit unexpectedly
            pyautogui.leftClick()

            pyautogui.keyUp('shift')
            pyautogui.keyDown('ctrl')
            pyautogui.press('left')
            pyautogui.press('left')
            sleep(0.5)
            pyautogui.keyUp('ctrl')

            # rejoin game CHANGE
            rejoin(website)

            # checks if joined game
            breakout = False
            for i in range(20):
                sleep(1)
                r, g, b, y = safe_pixel(340, 693) # checks if there is "skip intro" button (if the skip intro button is grey)
                if 35 < r < 45 and 35 < g < 45 and 35 < b < 45:
                    sleep(7)
                    pyautogui.moveTo(369, 693) # clicks on "skip kintro" button
                    pyautogui.leftClick()
                    sleep(1.5)

                    pyautogui.moveTo(696, 396) # clicks on the big green button to join
                    pyautogui.leftClick()
                    sleep(3)

                    pyautogui.moveTo(770, 491) # clicks on "no" when it asks you to favorite game 
                    pyautogui.leftClick()
                    sleep(2)
                    
                    reset()
                    breakout = True
                    break
            if breakout == True:
                break

        pyautogui.keyDown('shift')

    if subscription is None: 
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
        pyautogui.moveTo(886, 380) # clicks on PC
        sleep(0.5)
        pyautogui.leftClick()
        sleep(4)
        r, g, b, y = safe_pixel(439, 251) # checks PC color (checks if the red part is red)

        if r != 217 or g != 87 or b != 84: # if it isnt try to rejoin
                    
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
            pyautogui.moveTo(1321, 391) # click on the middle of the first box
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
            pyautogui.moveTo(446, 326) # click on the pokemon on the 2nd slot
            pyautogui.mouseDown()
            row = max(1, min(state["row"], 5))
            column = max(1, min(state["column"], 6))
            sleep(1)

            # YOU NEED TO EDIT THIS PART. 
            x0, y0 = 629, 227 # first slot in a box
            x1, y1 = 1100, 590 # last slot in a box

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
        pyautogui.moveTo(1146, 95) # click close PC
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
