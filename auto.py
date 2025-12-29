import os
import subprocess

base_folder = r"C:\Users\joshan\Desktop\joshan_3rdsem\FOD\assignment\scrapping"

subprocess.run(["python", os.path.join(base_folder, "insta-scrap.py")])


for folder_name in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(folder_path):
      
        subprocess.run(["python", os.path.join(base_folder, "OPCLIP.py"), folder_path])

