# Acoustic Biomarkers for Depression Detection

This project is a depression-detection demo built from the RAVDESS speech dataset. It uses a trained SVM (RBF) model with 40-dimensional MFCC features averaged over time, then standardization via a saved scaler for inference.

## Project Highlights

- Binary classification: `Healthy` vs `Sad_Depressed`
- Feature pipeline: `librosa.load()` -> MFCC-40 -> mean over time -> `StandardScaler` -> `SVM (RBF)`
- Professional Streamlit GUI for audio upload and sample-based prediction
- Prediction confidence display, waveform preview, and class probability visualization

## Key Results

- Best model: `SVM (RBF)`
- Test accuracy: `81.48%`
- Cross-validation F1: `0.8098 ± 0.0357`
- Training accuracy: `91.43%`
- Feature type: `MFCC-40 (mean over time)`

## Files Included

- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `best_model.pkl` - trained classifier
- `scaler.pkl` - fitted scaler used during training
- `label_encoder.pkl` - label mapping for class names
- `model_metadata.json` - saved model summary and metrics
- `DSAI231103043_Depression_Detection_SP_Project (1).ipynb` - full notebook
- `DSAI231103043_Depression_Detection_SP_Project.html` - notebook export
- `DSAI231103043_Depression_Detection_Presentation.pptx` - presentation

## How to Run

1. Clone the repository.
2. Open a terminal in the project folder.
3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Start the app:

```powershell
python -m streamlit run app.py
```

## How to Use the App

1. Upload a `.wav` file, or choose a sample if the dataset folder is available locally.
2. Click `Analyze Audio`.
3. Review the predicted label, confidence score, waveform, and class probabilities.

## Dataset Note

The original dataset folder is `audio_speech_actors_01-24/`. It is not required for upload-based inference, but if you want the in-app sample browser to work, keep that folder locally.

## Important Note

This project is for academic/demo use only and is not a clinical diagnostic tool.
