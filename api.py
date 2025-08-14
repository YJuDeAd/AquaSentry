import argparse
import numpy as np
import librosa
import tensorflow as tf
import os

# --- Configuration ---
SAMPLE_RATE = 22050  # Sample rate of the audio.
N_MFCC = 32          # Number of Mel-frequency cepstral coefficients (MFCCs) to extract.
MAX_LEN = 32         # Maximum length of the MFCC feature vector.

# --- Class Labels ---
CLASS_NAMES = [
    "Atlantic Spotted Dolphin", "Bearded Seal", "Beluga, White Whale",
    "Bottlenose Dolphin", "Bowhead Whale", "Calm Waters", "Clymene Dolphin",
    "Common Dolphin", "False Killer Whale", "Fin, Finback Whale", "Fishes",
    "Fraser's Dolphin", "Grampus, Risso's Dolphin", "Harp Seal",
    "Humpback Whale", "Killer Whale", "Leopard Seal", "Long-finned Pilot Whale",
    "Melon Headed Whale", "Minke Whale", "Narwhal", "Northern Right Whale",
    "Pantropical Spotted Dolphin", "Ross Seal", "Rough-toothed Dolphin",
    "Ships", "Short-finned Pilot Whale", "Southern Right Whale", "Sperm Whale",
    "Spinner Dolphin", "Striped Dolphin", "Submarines", "Violent Waters",
    "Walrus", "Weddell Seal", "White-beaked Dolphin", "White-sided Dolphin"
]

def preprocess_audio(file_path):
    """
    Loads an audio file and preprocesses it into the format expected by the model.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        numpy.ndarray: The preprocessed audio data (MFCCs).
    """
    try:
        # 1. Load the audio file using librosa
        signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

        # 2. Extract MFCCs from the audio signal
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=N_MFCC)

        # 3. Pad or truncate the MFCCs to a fixed length (MAX_LEN)
        if mfccs.shape[1] > MAX_LEN:
            mfccs = mfccs[:, :MAX_LEN]
        else:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None

    return mfccs

def inspect_model(model_path):
    """
    Loads a model and prints its summary and output shape to help determine the number of classes.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"--- Inspecting Model: {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.\n")
        
        model.summary()
        
        output_shape = model.output_shape
        print(f"\nModel Output Shape: {output_shape}")
        
        if isinstance(output_shape, list):
            output_shape = output_shape[0]

        if len(output_shape) < 2:
             print("\nCould not determine the number of classes from the output shape.")
             return

        num_classes = output_shape[-1]
        print(f"\nThis model was trained to predict {num_classes} different classes.")
        print(f"Please ensure your CLASS_NAMES list in the script has exactly {num_classes} items.")

    except Exception as e:
        print(f"\nAn error occurred while inspecting the model: {e}")
        print("Please ensure you are providing a valid TensorFlow/Keras model file (.h5 or .keras).")


def predict_sound(model_path, audio_path):
    """
    Loads a trained model, preprocesses an audio file, and predicts the sound.

    Args:
        model_path (str): Path to the saved Keras model file (.keras or .h5).
        audio_path (str): Path to the input audio file.
    """
    # 1. Load the pre-trained Keras model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # 2. Preprocess the input audio file
    mfccs = preprocess_audio(audio_path)
    if mfccs is None:
        return

    # 3. Get the model's expected input shape
    try:
        expected_input_shape = model.input_shape
        if isinstance(expected_input_shape, list):
            expected_input_shape = expected_input_shape[0]
        
        if len(expected_input_shape) != 2:
             print(f"\nError: The model's input layer is not a standard Dense layer.")
             print(f"Expected a 2D input shape like (None, num_features), but got {expected_input_shape}.")
             return
        expected_features = expected_input_shape[1]
    except Exception as e:
        print(f"Could not determine model's expected input shape. Error: {e}")
        return

    # 4. Flatten the preprocessed data and check if its shape matches the model's expectation.
    flattened_mfccs = mfccs.flatten()
    
    if flattened_mfccs.shape[0] != expected_features:
        print(f"\n--- CRITICAL SHAPE MISMATCH ---")
        print(f"The model expects input data with {expected_features} features.")
        print(f"However, the current preprocessing settings (N_MFCC, MAX_LEN) produced {flattened_mfccs.shape[0]} features.")
        print(f"You MUST adjust the N_MFCC and MAX_LEN variables in the script to match the settings used when the model was trained.")
        print(f"The script cannot continue.")
        return

    # 5. Reshape for prediction (add batch dimension for the model)
    data_for_prediction = flattened_mfccs[np.newaxis, ...]

    # 6. Make a prediction
    predictions = model.predict(data_for_prediction)

    # 7. Get the index of the highest probability
    predicted_index = np.argmax(predictions[0])
    
    if predicted_index >= len(CLASS_NAMES):
        print(f"\nError: The model predicted class index {predicted_index}, but the CLASS_NAMES list only has {len(CLASS_NAMES)} items.")
        return

    # 8. Map the index to the corresponding class name
    predicted_label = CLASS_NAMES[predicted_index]

    print(f"\nPrediction complete.")
    print(f"The predicted sound is: '{predicted_label}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify or inspect a sound model.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file (e.g., sound_model.keras).")
    parser.add_argument("audio_path", nargs='?', default=None, type=str, help="Path to the input audio file (required for prediction).")
    parser.add_argument("--inspect", action="store_true", help="Inspect the model's output shape instead of predicting.")

    args = parser.parse_args()

    if args.inspect:
        inspect_model(args.model_path)
    elif args.audio_path:
        predict_sound(args.model_path, args.audio_path)
    else:
        print("Error: You must either provide an audio file for prediction or use the --inspect flag.")
        parser.print_help()
