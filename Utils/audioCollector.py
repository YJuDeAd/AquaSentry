

# ====== USER SETTINGS ======
links = [
    "https://example.com/folder1", 
    "https://example.com/folder2", 
    # Add more links here
]
file_extensions = (".wav", ".mp3", ".zip")  # File types to download
# ===========================

def download_files(base_url, download_folder):
    """
    Downloads all files from a given base URL that match the specified file extensions.
    Saves them into the provided download_folder.
    """
    # Create the download folder if it doesn't exist
    os.makedirs(download_folder, exist_ok=True)
    print(f"\n📥 Downloading from: {base_url}")

    # Fetch the HTML content of the given URL
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error fetching {base_url}: {e}")
        return

    # Parse HTML to find file links
    soup = BeautifulSoup(response.text, "html.parser")
    found_files = 0 

    # Loop through all hyperlinks on the page
    for link in soup.find_all("a", href=True):
        file_url = urljoin(base_url, link["href"])
        if file_url.lower().endswith(file_extensions):
            file_name = os.path.basename(file_url)
            file_path = os.path.join(download_folder, file_name)

            # Attempt to download the file
            try:
                print(f"⬇ Downloading: {file_name}")
                file_data = requests.get(file_url)
                file_data.raise_for_status()

                with open(file_path, "wb") as f:
                    f.write(file_data.content)
                found_files += 1
            except Exception as e:
                print(f"❌ Failed to download {file_url}: {e}")

    # Summary of downloads
    if found_files == 0:
        print("⚠ No matching files found.")
    else:
        print(f"✅ Download complete. {found_files} files saved in {download_folder}")

def generate_spectrograms(wav_folder):
    """
    Generates spectrogram images for all .wav files in the given folder.
    Saves spectrograms in a 'spectrograms'.
    """
    # Create spectrogram output folder
    output_folder = os.path.join(wav_folder, "spectrograms")
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the folder
    for filename in os.listdir(wav_folder):
        if filename.lower().endswith(".wav"):  # Only process WAV files
            wav_path = os.path.join(wav_folder, filename)
            try:
                sample_rate, samples = wavfile.read(wav_path)
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
                continue

            # If stereo audio, use only the first channel
            if samples.ndim > 1:
                samples = samples[:, 0]

            # Create a spectrogram plot
            plt.figure(figsize=(10, 6))
            plt.specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=512, cmap="inferno")
            plt.title(f"Spectrogram - {filename}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency (Hz)")
            plt.colorbar(label="Intensity (dB)")

            # Save spectrogram image
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()

            print(f"📊 Saved spectrogram: {output_path}")

# ===== MAIN PROCESS =====
# For each link, create a numbered download folder and process files
for i, url in enumerate(links, start=1):
    download_folder = os.path.join("downloads", str(i))
    download_files(url, download_folder)  # Step 1: Download files
    generate_spectrograms(download_folder)  # Step 2: Generate spectrograms

print("\n🎉 All downloads & spectrograms complete!")
