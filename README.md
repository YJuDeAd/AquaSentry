# AquaSentry 🌊🛡️

**Status:** 🚧 In Planning & Development 🚧

**AquaSentry** is a project aimed at building an AI-powered surveillance system for **Underwater Domain Awareness**. Our goal is to analyze hydrophone audio data to automatically detect, analyze, and classify acoustic anomalies.

This project is being built for the [House of Turning](https://www.techtatva.club/house-of-turing).

---

## The Vision

The underwater domain is filled with ambient noise, making it difficult to manually monitor for specific objects of interest (like ships, submarines, or even marine life). Our vision for **AquaSentry** is to create an intelligent system that acts as a vigilant "sentry," listening to the ocean's sounds and alerting users to significant events.

---

## Project Goals

Our primary objectives are to develop a system that can:

1.  **Isolate Anomalies:** Successfully distinguish meaningful signals from the background ambient noise in an audio file.
2.  **Characterize Objects:** Extract a set of unique acoustic features (a "signature") from each detected anomaly.
3.  **Classify Threats:** Build a machine learning model to classify these signatures into predefined categories.

---

## Proposed Technical Approach

We plan to implement a multi-stage pipeline:

1.  **Signal Processing:** Load `.wav` files and apply filters to clean the audio. We will establish a baseline for ambient noise.
2.  **Event Detection:** Implement an algorithm to scan the audio and segment out clips where the sound energy surpasses the baseline noise profile.
3.  **Feature Extraction:** For each event, we plan to extract features like MFCCs, spectral contrast, and chroma.
4.  **Classification:** Use the extracted features to train a classifier (e.g., SVM, RandomForest, or a small Neural Network) to identify the object's class.

---

## Hackathon Roadmap

We will be tracking our progress with the following milestones:

- [ ] **Milestone 1: Data Handling & Pre-processing**
  - [ ] Script to load and process `.wav` files.
  - [ ] Implement a noise reduction/profiling algorithm.
- [ ] **Milestone 2: Anomaly Detection**
  - [ ] Develop the core logic for detecting and segmenting audio events.
  - [ ] Test detection on sample audio clips.
- [ ] **Milestone 3: Feature Engineering & Classification**
  - [ ] Build the feature extraction pipeline.
  - [ ] Train an initial classification model.
- [ ] **Milestone 4: Integration & Demo**
  - [ ] Combine all parts into a single, executable script.
  - [ ] Prepare a final presentation and demonstration.

---

## Proposed Technology Stack

Our stack is organized by function to tackle this challenge effectively.

* **Deep Learning & Transformers (Hugging Face Ecosystem)**
    * `torch`, `torchvision`, `torchaudio`: The core PyTorch framework for building models.
    * `transformers`, `datasets`: For accessing and fine-tuning state-of-the-art models.
    * `peft`, `trl`: For efficient fine-tuning techniques (Parameter-Efficient Fine-Tuning).
    * `huggingface_hub`: For model and dataset sharing.

* **Computer Vision on Audio**
    * `ultralytics`: To use YOLO models for detecting events in spectrograms.
    * `Pillow`, `matplotlib`: For creating and manipulating spectrogram images.

* **Classical ML & Data Handling**
    * `scikit-learn`: For traditional ML models and data preprocessing.
    * `xgboost`: A powerful gradient-boosting model for baseline comparison.
    * `pandas`: For organizing features and results.

* **Core Audio Processing**
    * `librosa`: The primary tool for audio feature extraction and analysis.

---
## Getting Started

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/YJuDeAd/AquaSentry
    cd AquaSentry
    ```

2.  **Install dependencies:**
    ```sh
    pip install transformers datasets peft trl torch torchvision torchaudio tensorflow Pillow matplotlib librosa pandas scikit-learn xgboost ultralytics huggingface_hub
    ```

---

## Contributors

* [Punya Arora](https://github.com/YJuDeAd)
* [Prasun Jha](https://github.com/PrasunJha15)
* [Rajmangalam Gupta](https://github.com/RajmangalmGupta)
* [Vaibhav Rustagi](https://github.com/getit-pajji)