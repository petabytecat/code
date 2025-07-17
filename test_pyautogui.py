import pyautogui
import time

# Safety feature - abort if mouse moves to corner
pyautogui.FAILSAFE = True

def control_keyboard_mouse():
    # Give user time to position window
    time.sleep(3)
    
    # Example 1: Click and type
    pyautogui.click(500, 300)  # Click at (x=500, y=300)
    pyautogui.typewrite("Hello World!", interval=0.1)
    
    # Example 2: Press special keys
    pyautogui.press('enter')
    pyautogui.hotkey('ctrl', 'c')  # Copy command
    
    # Example 3: Right-click context menu
    pyautogui.rightClick(700, 400)
    time.sleep(1)
    pyautogui.press('down')  # Navigate menu
    pyautogui.press('enter')

    # Example 4: Double-click and drag
    pyautogui.doubleClick(200, 150)
    pyautogui.dragTo(300, 250, button='left')

# Get current mouse position (debugging)
print(f"Current position: {pyautogui.position()}")

# Execute the sequence
control_keyboard_mouse()