import subprocess
import time
import os
import signal
import random

def get_random_quote():
    try:
        with open('quotes.txt', 'r') as file:
            quotes = file.read().splitlines()
            # Remove empty lines
            quotes = [quote for quote in quotes if quote.strip()]
            return random.choice(quotes)
    except Exception as e:
        return f"Error reading quotes: {e}"

def format_quote(quote):
    width = 60
    border = "=" * width
    padding = " " * 2

    return f"\n{border}\n{padding}{quote}\n{border}\n"

def are_apps_running(app_name):
    try:
        output = subprocess.check_output(["pgrep", "-f", app_name])
        return True if output else False
    except subprocess.CalledProcessError:
        return False

def kill_app(app_name):
    try:
        subprocess.call(["pkill", "-f", app_name])
        quote = get_random_quote()
        print(format_quote(quote))
        #print(f"Terminated {app_name}")
    except Exception as e:
        print(f"Error terminating {app_name}: {e}")

def block_all_apps(app_list=["Google Chrome", "Lunar Client", "Roblox", "Safari"]):
    for app in app_list:
        if are_apps_running(app):
            kill_app(app)

# Ignore SIGINT (Ctrl+C)
def signal_handler(sig, frame):
    pass  # Do nothing on SIGINT

signal.signal(signal.SIGINT, signal_handler)

while True:
    time.sleep(10)
    block_all_apps()
