import sys
import librosa
import numpy as np
import tensorflow as tf

# Load the model weights you saved earlier
model = tf.keras.models.load_model('speech_emotion_model.h5')

def run_inference(audio_path):
    try:
        # Preprocessing: Load 3s of audio and convert to 128x128 Mel-spectrogram
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Ensure the dimensions are exactly 128x128
        if mel_db.shape[1] > 128:
            mel_db = mel_db[:, :128]
        elif mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
            
        # Reshape for CNN input (Batch, Height, Width, Channels)
        input_data = mel_db.reshape(1, 128, 128, 1)
        
        # Predict and calculate confidence percentage
        predictions = model.predict(input_data, verbose=0)
        confidence = np.max(predictions) * 100
        emotion_idx = np.argmax(predictions)
        
        emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
        
        print("\n" + "="*30)
        print(f"RESULT: {emotions[emotion_idx].upper()}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("="*30)
        
    except Exception as e:
        print(f"Error processing audio file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        print("Usage: python predict.py <path_to_wav_file>")
