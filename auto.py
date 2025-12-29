import os
import subprocess

# Path to your scrapping folder
base_folder = r"C:\Users\joshan\Desktop\joshan_3rdsem\FOD\assignment\scrapping"

# Step 1: Run insta-scrap.py to download images
subprocess.run(["python", os.path.join(base_folder, "insta-scrap.py")])

# Step 2: Analyze downloaded folders (folders named as usernames)
for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path):
        # Send folder to OPCLIP.py for analysis
        subprocess.run(["python", os.path.join(base_folder, "OPCLIP.py"), folder_path])
