import os
import re

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Extract the components using regex
            match = re.match(r'(\d{4}_RANGER)_(.+)_-_(.+)_Spec_Sheet_(rank\d+)_(page\d+\.png)', filename)

            if match:
                year_explorer, university, team, rank, page = match.groups()

                # Create the new filename
                new_filename = f"{year_explorer}_{rank}_{university}_-_{team}_Spec_Sheet_{page}"

                # Rename the file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed: {filename} -> {new_filename}")

# Usage
directory = 'spec_sheets_pngs'  # Replace with the actual path to your folder
rename_files(directory)
