import os

# --- Configuration ---
parent_folder = r"C:\Example" # Replace with the actual video link

# --- 1. Get all subfolders inside the parent folder ---
subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]

# --- 2. Loop through each subfolder ---
for subfolder in subfolders:
    subfolder_path = os.path.join(parent_folder, subfolder)
    
    # --- 3. List all files in the current subfolder ---
    files = os.listdir(subfolder_path)
    
    # --- 4. Loop through each file in the subfolder with a counter ---
    for index, file in enumerate(files, start=1):
        file_path = os.path.join(subfolder_path, file)
        
        # Skip if the item is a directory (only rename files)
        if os.path.isdir(file_path):
            continue
        
        # --- 5. Get the file extension ---
        _, ext = os.path.splitext(file)
        
        # --- 6. Create the new filename ---
        new_name = f"{subfolder}_{index}{ext}"
        new_path = os.path.join(subfolder_path, new_name)
        
        # --- 7. Rename the file ---
        os.rename(file_path, new_path)
        print(f"Renamed: {file_path} -> {new_path}")

# --- 8. Done ---
print("✅ Renaming complete.")
