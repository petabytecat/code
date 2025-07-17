from time import sleep
import pyautogui
import getpixelcolor
from PIL import ImageGrab

sleep(2)

while True:
    print("Unable to join for 5 rounds...")
    
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


