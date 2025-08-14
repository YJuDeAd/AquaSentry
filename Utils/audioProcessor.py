import os
import librosa
import soundfile as sf
from glob import glob

# --- Configuration ---
# The parent folder containing all your audio files and subfolders.
PARENT_SOURCE_DIR = r'C:\Example' # Replace with the actual video link
# A new folder where the correctly formatted audio will be saved.
DEST_AUDIO_DIR = 'processed_audio'
# The sample rate YAMNet expects.
TARGET_SR = 16000

print("Starting audio preparation...")

# --- 1. Check if the parent source directory exists ---
if not os.path.isdir(PARENT_SOURCE_DIR):
    print(f"FATAL ERROR: Parent source directory not found at '{PARENT_SOURCE_DIR}'")
    exit()

# --- 2. Create the main destination directory ---
if not os.path.exists(DEST_AUDIO_DIR):
    os.makedirs(DEST_AUDIO_DIR)
    print(f"Created destination directory: {DEST_AUDIO_DIR}")

# --- 3. Find all .wav files recursively ---
print(f"Searching for all .wav files in '{PARENT_SOURCE_DIR}' and its subfolders...")
try:
    search_pattern = os.path.join(PARENT_SOURCE_DIR, '**', '*.wav')
    audio_files = glob(search_pattern, recursive=True)
except Exception as e:
    print(f"Error finding audio files: {e}")
    exit()

if not audio_files:
    print(f"Warning: No .wav files were found in '{PARENT_SOURCE_DIR}'.")
    exit()

print(f"Found a total of {len(audio_files)} audio files to process.")
total_files_processed = 0

# --- 4. Loop through each found audio file and process it ---
for filepath in audio_files:
    try:
        # 'mono=True' converts it to a single channel.
        wav, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Resample the audio to the target sample rate (16kHz) if it's not already.
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        
        relative_path = os.path.relpath(filepath, PARENT_SOURCE_DIR)
        dest_path = os.path.join(DEST_AUDIO_DIR, relative_path)
        
        # Create the destination subfolder if it doesn't exist.
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Save the newly formatted audio file.
        sf.write(dest_path, wav, TARGET_SR)
        total_files_processed += 1
        
    except Exception as e:
        print(f"Could not process {os.path.basename(filepath)}. Error: {e}")

print(f"\n--------------------------------------------------")
print(f"✅ Audio preparation complete!")
print(f"Total files processed: {total_files_processed}")
print(f"All formatted files are now in the '{DEST_AUDIO_DIR}' folder.")
print(f"--------------------------------------------------")