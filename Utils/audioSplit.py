import os
from pydub import AudioSegment

# --- Configuration ---
# Path to your main folder that contains WAV files (including in subfolders)
main_folder = r"C:\Example" # Replace with the actual video link

# Desired segment length in milliseconds (7 seconds = 7 * 1000 ms)
segment_length = 7 * 1000  # milliseconds

# --- 1. Walk through all subdirectories in the main folder ---
for root, dirs, files in os.walk(main_folder):
    for file in files:
        # Process only files with ".wav" extension (case-insensitive)
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            print(f"Processing: {file_path}")
            
            # --- 2. Load the WAV file ---
            audio = AudioSegment.from_wav(file_path)
            duration_ms = len(audio)  
            
            # --- 3. Split into chunks of 7 seconds each ---
            part_num = 1 
            for start in range(0, duration_ms, segment_length):
                # End position is either 7 seconds ahead or the end of the file
                end = min(start + segment_length, duration_ms)
                chunk = audio[start:end]
                
                # Skip saving if chunk is empty (shouldn’t normally happen)
                if len(chunk) == 0:
                    continue
                
                # --- 4. Create the new filename ---
                base_name = os.path.splitext(file)[0] 
                new_filename = f"{base_name}_part{part_num}.wav"
                new_filepath = os.path.join(root, new_filename) 
                
                # --- 5. Export the chunk as a WAV file ---
                chunk.export(new_filepath, format="wav")
                part_num += 1

# --- 6. Done ---
print("✅ Splitting complete!")
