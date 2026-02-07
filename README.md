# Speech Emotion Recognition (SER)

**Submitted by:** Aarya Mehrotra  
**Student ID:** 2025B3PS1042P

## 1. Model Performance
* **Test Accuracy:** 27%
* **Macro F1-Score:** 0.17
* **Target Emotions:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised.

## 2. Training Details
* **Architecture:** 2D Convolutional Neural Network (CNN).
* **Input Shape:** 128x128 Mel-spectrograms.
* **Data Split:** 80% Training, 10% Validation, 10% Testing (Stratified).
* **Data Augmentations Used:** * **Noise Injection:** Added white noise to audio.
    * **Pitch Shifting:** Changed pitch without changing duration.
    * **Time Stretching:** Changed speed of audio.
* **Optimizer:** Adam.
* **Epochs:** 50.

## 3. How to Run Inference (Phase 3)
To test a single audio file and see the **Emotion** and **Confidence Percentage**, run the following command in your terminal:

```bash
python predict.py "path/to/your/audio_file.wav"
