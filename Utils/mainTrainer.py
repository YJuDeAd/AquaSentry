import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# --- GPU Diagnostics ---
print("--- GPU DIAGNOSTICS ---")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"✅ Found {len(gpu_devices)} GPU(s): {gpu_devices}")
    print("TensorFlow should be using the GPU.")
else:
    print("❌ TensorFlow could not find any GPUs.")
    print("This is why the script is running on the CPU.")
print("-----------------------")


# --- Configuration ---
# The folder containing your processed audio from the previous script.
PROCESSED_AUDIO_DIR = 'processed_audio'
# The handle for the pre-trained YAMNet model on TensorFlow Hub.
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
# Filepath to save the best model
SAVED_MODEL_PATH = 'best_classifier.keras'

# --- 1. Load the YAMNet Model ---
print("Loading YAMNet model from TensorFlow Hub...")
yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
print("YAMNet model loaded.")

# --- Helper Function to Load Audio ---
def load_wav_16k_mono(filename):
    """Load a WAV file, convert it to mono, and resample to 16 kHz."""
    wav, _ = librosa.load(filename, sr=16000, mono=True)
    return wav

# --- 2. Prepare File List and Labels ---
print("Preparing file list and labels by recursively scanning for .wav files...")
filepaths = []
labels = []

search_pattern = os.path.join(PROCESSED_AUDIO_DIR, '**', '*.wav')
for filepath in glob.glob(search_pattern, recursive=True):
    filepaths.append(filepath)
    label = os.path.basename(os.path.dirname(filepath))
    labels.append(label)

if not filepaths:
    print(f"FATAL ERROR: No .wav files found in '{PROCESSED_AUDIO_DIR}'.")
    exit()

class_names = sorted(list(set(labels)))
print(f"Found {len(filepaths)} files belonging to {len(class_names)} classes.")

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
le = LabelEncoder()
# Make sure the encoder learns all possible class names
le.fit(class_names)
df['label_id'] = le.transform(df['label'])

id_to_class_name = {i: name for i, name in enumerate(le.classes_)}
print(f"Labels encoded into {len(le.classes_)} unique IDs.")

# --- 3. Split Data Before Extracting Embeddings ---
# This is important so we only augment the training data.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_id'])
print(f"Data split into {len(train_df)} training samples and {len(test_df)} testing samples.")

# --- 4. Define Augmentation Pipeline ---
# This will create slightly modified versions of our audio.
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# --- 5. Extract Embeddings and Apply Augmentation ---
def extract_embeddings(dataframe, should_augment):
    """Extracts YAMNet embeddings. Applies augmentation if specified."""
    embeddings_list = []
    labels_list = []
    for i, row in dataframe.iterrows():
        try:
            wav_data = load_wav_16k_mono(row['filepath'])
            
            # Extract embedding for the original audio
            _, embeddings, _ = yamnet_model(wav_data)
            embeddings_list.append(np.mean(embeddings.numpy(), axis=0))
            labels_list.append(row['label_id'])
            
            # Apply augmentation ONLY to the training data
            if should_augment:
                for _ in range(2): # Create 2 augmented versions per file
                    augmented_wav = augmenter(samples=wav_data, sample_rate=16000)
                    _, aug_embeddings, _ = yamnet_model(augmented_wav)
                    embeddings_list.append(np.mean(aug_embeddings.numpy(), axis=0))
                    labels_list.append(row['label_id']) # Label remains the same

        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")

    return np.array(embeddings_list), np.array(labels_list)

print("\nExtracting embeddings for the training set (with augmentation)...")
X_train, y_train = extract_embeddings(train_df, should_augment=True)
print(f"Training set size after augmentation: {len(X_train)} samples.")

print("\nExtracting embeddings for the test set (no augmentation)...")
X_test, y_test = extract_embeddings(test_df, should_augment=False)

# --- 6. Balance the Training Data with SMOTE ---
print("\nBalancing the training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training set size after SMOTE: {len(X_train_resampled)} samples.")


# --- 7. Build a Deeper Model ---
print("\nBuilding the classifier model...")
num_classes = len(class_names)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name='input_embedding'),
    
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation='softmax')
], name='fish_classifier_v3')

model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Slightly lower learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 8. Define Callbacks and Train the Model ---
print("\n--- Starting Model Training ---")
# Define callbacks for more advanced training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=7, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(SAVED_MODEL_PATH, monitor='val_accuracy', save_best_only=True)
]

history = model.fit(X_train_resampled, y_train_resampled, # Use the resampled data
                    epochs=150,  # Increased epochs, EarlyStopping will handle the rest
                    batch_size=32,
                    validation_data=(X_test, y_test), # Validate on the original test set
                    callbacks=callbacks)

print("--- Model Training Complete ---")

# --- 9. Load Best Model and Generate Classification Report ---
print(f"\n--- Loading best saved model from '{SAVED_MODEL_PATH}' and generating report ---")
best_model = tf.keras.models.load_model(SAVED_MODEL_PATH)
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"\nBest Model Test Accuracy: {accuracy * 100:.2f}%")
print(f"Best Model Test Loss: {loss:.4f}")

# Generate detailed report
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
# We explicitly provide the full range of labels to ensure the report includes all classes.
report_labels = np.arange(len(le.classes_))
print(classification_report(y_test, y_pred_classes, target_names=le.classes_, labels=report_labels, zero_division=0))

# --- 10. Generate and Display Confusion Matrix ---
print("\nGenerating confusion matrix...")
# This is a more robust way to create the confusion matrix plot that avoids the error.
# First, we compute the matrix itself, making sure to specify all possible labels.
cm_labels = np.arange(len(class_names))
cm = confusion_matrix(y_test, y_pred_classes, labels=cm_labels)

# Then, we create the display object from the pre-computed matrix.
fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

# Finally, we plot it.
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')

ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.show()


print("\nScript finished.")
