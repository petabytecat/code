import subprocess
import os
import shutil
from datetime import datetime

# Run the AppleScript
subprocess.run(["osascript", "import_photos.scpt"])

# Optional: move & organize imported files
source_dir = "/Users/dewei.zhang/Desktop/coding/import_phone_images"
target_dir = "/Volumes/David/___Storage___/pics/2019-2024_huoming_iphonexr"

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.mov')):
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved {filename}")
