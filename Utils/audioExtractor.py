import subprocess
import os

# --- Configuration ---
# URL of the video to download
url = "https://example.com/video"  # Replace with the actual video link
# Folder where the downloaded audio file will be saved
output_folder = r"C:\Example"  # Change this path to your desired location
# Name for the output WAV file
wav_filename = "output.wav"

# --- 1. Ensure the output folder exists ---
os.makedirs(output_folder, exist_ok=True)

# --- 2. Run yt-dlp to download and convert the video into WAV format ---
subprocess.run([
    "yt-dlp",                             
    "-x",                                 
    "--audio-format", "wav",              
    "-o", os.path.join(output_folder, wav_filename),  
    url                                  
])