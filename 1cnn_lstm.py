import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# Function to extract features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Extract Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Combine features
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ])
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Base path for dataset
base_path = r"C:\Users\Hp\PycharmProjects\Sitar_Script\Trimmed"
features_list = []
labels_list = []

# Loop through folders and process audio files
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        label = folder_name  # Use full folder name as label
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                features = extract_features(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)

# Convert features and labels to numpy arrays
X = np.array(features_list)
y = np.array(labels_list)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# Reshape for CNN + LSTM input (add channels dimension)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build CNN + LSTM Model
model = Sequential([
    # 1D Convolutional Layer
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    Dropout(0.3),
    # Reshape the output to match the expected shape for LSTM
    Reshape((X_train.shape[1]-4, 128)),  # Adjust dimensions to fit LSTM (timesteps, features)
    # LSTM Layer
    Bidirectional(LSTM(128, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_sitar.keras', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stop, model_checkpoint])

# Save the final model in .keras format
model.save('sitar_note_cnnlstm.keras')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test data
predictions = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Print predicted and true labels for all samples
true_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))
for true, pred in zip(true_labels, predicted_labels):
    print(f"True Label: {true}, Predicted Label: {pred}")
