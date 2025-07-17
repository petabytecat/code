import time
import pyautogui
import platform
import random

time.sleep(3)

system_name = platform.system()
modifier = 'command' if system_name == 'Darwin' else 'ctrl'
pyautogui.FAILSAFE = True

total_hours = 23
total_duration_seconds = total_hours * 3600
interval_seconds = 150  # 2 minutes 5 seconds

try:
    start_time = time.time()
    next_paste_time = start_time

    print(f"Starting clipboard paste every 2 minutes 5 seconds for {total_hours} hours.")
    print("Press Ctrl+C to stop early or move mouse to top-left corner to abort.")

    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= total_duration_seconds:
            print("Completion time reached. Exiting.")
            break

        if current_time < next_paste_time:
            wait_time = next_paste_time - current_time
            if wait_time > (total_duration_seconds - elapsed_time):
                break
            time.sleep(wait_time)
        
        # Paste clipboard content
        pyautogui.hotkey(modifier, 'v')
        
        # Randomly add punctuation 30% of the time
        if random.random() < 0.3:
            # Choose random punctuation (dot or comma)
            punctuation = random.choice([',', '.'])
            # Type the punctuation
            pyautogui.write(punctuation, interval=0.05)
            # Immediately delete it
            #pyautogui.press('backspace')
            # Add small random delay before pressing enter
            #time.sleep(random.uniform(0.05, 0.2))
        
        # Press Enter to send
        pyautogui.press("enter")
        
        print(f"Pasted at {time.strftime('%H:%M:%S')} - Hour {elapsed_time/3600:.2f}/23")
        next_paste_time += interval_seconds

except KeyboardInterrupt:
    print("\nProgram stopped by user.")
except pyautogui.FailSafeException:
    print("\nFailsafe triggered (mouse moved to top-left corner).")