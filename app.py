from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIRS = [APP_DIR, APP_DIR / "saved_models"]


def resolve_artifact(filename: str) -> Path:
    for base_dir in ARTIFACT_DIRS:
        candidate = base_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in the app folder or saved_models/")


@st.cache_resource
def load_artifacts():
    model_path = resolve_artifact("best_model.pkl")
    scaler_path = resolve_artifact("scaler.pkl")
    label_encoder_path = resolve_artifact("label_encoder.pkl")
    metadata_path = resolve_artifact("model_metadata.json")

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with open(label_encoder_path, "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return model, scaler, label_encoder, metadata


@st.cache_data
def discover_sample_files(limit: int = 18) -> list[str]:
    dataset_dir = APP_DIR / "audio_speech_actors_01-24"
    if not dataset_dir.exists():
        return []

    wav_files = sorted(str(path) for path in dataset_dir.glob("Actor_*/*.wav"))
    if len(wav_files) <= limit:
        return wav_files

    selected = []
    actor_buckets: dict[str, list[str]] = {}
    for file_path in wav_files:
        actor = Path(file_path).parent.name
        actor_buckets.setdefault(actor, []).append(file_path)

    for actor in sorted(actor_buckets.keys()):
        selected.append(actor_buckets[actor][0])
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for file_path in wav_files:
            if file_path not in selected:
                selected.append(file_path)
            if len(selected) >= limit:
                break

    return selected[:limit]


def infer_label_from_filename(file_path: str) -> str:
    file_name = Path(file_path).name
    parts = file_name.split("-")
    if len(parts) < 3:
        return "Unknown"

    emotion_code = parts[2]
    if emotion_code in {"01", "02"}:
        return "Healthy"
    if emotion_code == "04":
        return "Sad_Depressed"
    return "Excluded"


def group_samples_by_label(file_paths: list[str]) -> dict[str, list[str]]:
    grouped = {"Healthy": [], "Sad_Depressed": [], "Excluded": []}
    for file_path in file_paths:
        grouped.setdefault(infer_label_from_filename(file_path), []).append(file_path)
    return grouped


def extract_features(audio_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    audio, sample_rate = librosa.load(audio_path)
    mfcc_vector = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return audio, mfcc_vector, sample_rate


def predict_audio(audio_path: str, model, scaler, label_encoder):
    audio, mfcc_vector, sample_rate = extract_features(audio_path)
    mfcc_2d = mfcc_vector.reshape(1, -1)
    scaled_features = scaler.transform(mfcc_2d)

    prediction_code = model.predict(scaled_features)[0]
    predicted_label = label_encoder.inverse_transform([prediction_code])[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        probability_map = dict(zip(label_encoder.classes_, probabilities))
        confidence = float(probabilities[prediction_code])
    else:
        score = float(model.decision_function(scaled_features)[0])
        confidence = abs(score) / (abs(score) + 1.0)
        probability_map = {predicted_label: confidence}

    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probability_map": probability_map,
    }


def build_waveform_figure(audio: np.ndarray, sample_rate: int, predicted_label: str):
    duration = len(audio) / sample_rate
    time_axis = np.linspace(0, duration, num=len(audio))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=audio,
            mode="lines",
            line=dict(color="#1D4ED8", width=1.5),
            name="Waveform",
        )
    )
    fig.update_layout(
        title="Waveform Preview",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
    )
    if predicted_label == "Sad_Depressed":
        fig.update_traces(line=dict(color="#B91C1C", width=1.5))
    return fig


def build_probability_figure(probability_map: dict[str, float], predicted_label: str):
    classes = list(probability_map.keys())
    values = [probability_map[name] for name in classes]
    colors = ["#0F766E" if label == "Healthy" else "#B91C1C" for label in classes]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker_color=colors,
            text=[f"{value * 100:.1f}%" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Prediction Confidence",
        xaxis=dict(title="Probability", range=[0, 1]),
        yaxis=dict(title="Class"),
        template="plotly_white",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#64748B")
    if predicted_label == "Sad_Depressed":
        fig.update_traces(marker_color=["#1D4ED8" if label == "Healthy" else "#B91C1C" for label in classes])
    return fig


def build_performance_figure(metadata: dict):
    labels = ["Train Accuracy", "Test Accuracy", "CV F1 Mean", "Overfit Gap"]
    values = [
        float(metadata.get("train_accuracy", 0.0)),
        float(metadata.get("test_accuracy", 0.0)),
        float(metadata.get("cv_f1_mean", 0.0)),
        float(metadata.get("overfit_gap", 0.0)),
    ]
    colors = ["#0F766E", "#1D4ED8", "#7C3AED", "#F59E0B"]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.3f}" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model Performance Snapshot",
        yaxis=dict(title="Score", range=[0, 1.05]),
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
    )
    return fig


def app_css() -> str:
    return """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 216, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(13, 148, 136, 0.16), transparent 26%),
                linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
            color: #0f172a;
        }
        .hero {
            padding: 1.8rem 1.8rem 1.2rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.93));
            color: white;
            box-shadow: 0 22px 60px rgba(15, 23, 42, 0.25);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.7rem 0 0 0;
            color: rgba(255,255,255,0.82);
            font-size: 1.0rem;
        }
        .soft-note {
            background: rgba(255,255,255,0.75);
            border-left: 4px solid #1d4ed8;
            padding: 0.9rem 1rem;
            border-radius: 12px;
            margin-top: 0.8rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.4rem 0.8rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .status-healthy { background: rgba(16, 185, 129, 0.12); color: #047857; }
        .status-sad { background: rgba(239, 68, 68, 0.12); color: #b91c1c; }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            margin: 0 0 0.5rem 0;
            color: #0f172a;
        }
    </style>
    """


def display_result(result: dict, label_encoder):
    predicted_label = result["predicted_label"]
    confidence = result["confidence"]
    probability_map = result["probability_map"]

    status_class = "status-sad" if predicted_label == "Sad_Depressed" else "status-healthy"
    status_title = "Sad / Depressed" if predicted_label == "Sad_Depressed" else "Healthy"

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Class", status_title)
    col2.metric("Confidence", f"{confidence * 100:.1f}%")
    col3.metric("Classes", ", ".join(label_encoder.classes_))

    st.markdown(
        f'<div class="soft-note"><span class="status-pill {status_class}">{status_title}</span> '
        f'Prediction completed using the trained MFCC-40 SVM pipeline.</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.plotly_chart(build_waveform_figure(result["audio"], result["sample_rate"], predicted_label), use_container_width=True)
    with right_col:
        st.plotly_chart(build_probability_figure(probability_map, predicted_label), use_container_width=True)


def render_evaluation_dashboard(metadata: dict):
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(build_performance_figure(metadata), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Model facts</div>', unsafe_allow_html=True)
        st.markdown(
            f"- Project: {metadata.get('project', 'Acoustic Biomarkers for Depression Detection')}\n"
            f"- Best model: {metadata.get('best_model_name', 'SVM (RBF)')}\n"
            f"- Model class: {metadata.get('model_class', 'SVC')}\n"
            f"- Dataset: {metadata.get('dataset', 'RAVDESS')}\n"
            f"- Feature type: {metadata.get('feature_type', 'MFCC-40 (mean over time)')}\n"
            f"- Classes: {len(metadata.get('classes', []))}"
        )
        gap = float(metadata.get("overfit_gap", 0.0))
        if gap <= 0.08:
            st.success("Generalization looks healthy for a class project.")
        else:
            st.warning("There is a visible train-test gap, so avoid overclaiming clinical readiness.")


def main():
    st.set_page_config(page_title="Depression Detection Studio", page_icon="🎧", layout="wide")
    st.markdown(app_css(), unsafe_allow_html=True)

    model, scaler, label_encoder, metadata = load_artifacts()
    sample_files = discover_sample_files()
    grouped_samples = group_samples_by_label(sample_files)

    with st.sidebar:
        st.markdown("## Depression Detection Studio")
        st.caption("Acoustic biomarker demo for your SP project")
        st.markdown(
            f"**Model:** {metadata.get('best_model_name', 'SVM (RBF)')}\n\n"
            f"**Accuracy:** {metadata.get('test_accuracy', 0) * 100:.2f}%\n\n"
            f"**Feature:** {metadata.get('feature_type', 'MFCC-40')}"
        )
        st.divider()
        st.markdown("### Sample browser")
        source_choice = st.radio("Source", ["Upload WAV", "Pick sample"], index=0)
        label_choice = st.selectbox("Filter by label", ["All", "Healthy", "Sad_Depressed", "Excluded"])
        st.divider()
        st.markdown("### Notes")
        st.info("Uploaded files are processed in-memory and deleted after analysis.")

    st.markdown(
        """
        <div class="hero">
            <h1>Depression Detection Studio</h1>
            <p>Professional acoustic biomarker demo built from your RAVDESS-based MFCC-40 SVM project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    meta_cols = st.columns(4)
    meta_cols[0].metric("Best Model", metadata.get("best_model_name", "SVM (RBF)"))
    meta_cols[1].metric("Test Accuracy", f"{metadata.get('test_accuracy', 0) * 100:.2f}%")
    meta_cols[2].metric("CV F1", f"{metadata.get('cv_f1_mean', 0):.4f}")
    meta_cols[3].metric("Feature Type", metadata.get("feature_type", "MFCC-40"))

    tab_predict, tab_dashboard, tab_about, tab_workflow = st.tabs(["Predict Audio", "Evaluation", "Project Info", "How It Works"])

    with tab_predict:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Upload or choose a sample")
            uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"] if source_choice == "Upload WAV" else ["wav"], disabled=source_choice != "Upload WAV")
            selected_sample = None
            filtered_samples = sample_files
            if label_choice != "All":
                filtered_samples = grouped_samples.get(label_choice, [])

            if source_choice == "Pick sample" and filtered_samples:
                selected_sample = st.selectbox(
                    "Or choose a dataset sample",
                    ["-- choose a sample --"] + filtered_samples,
                    index=0,
                    format_func=lambda value: Path(value).name if value != "-- choose a sample --" else value,
                )
            elif source_choice == "Pick sample" and not filtered_samples:
                st.warning("No samples found for the selected label filter.")

            predict_clicked = st.button("Analyze Audio", type="primary", use_container_width=True)
            st.info("The model expects a WAV file and uses the same 40 MFCC mean-vector pipeline as your notebook.")

        with right:
            st.markdown("### Model snapshot")
            st.markdown(
                f"- Model: {metadata.get('model_class', type(model).__name__)}\n"
                f"- Dataset: {metadata.get('dataset', 'RAVDESS')}\n"
                f"- Classes: {', '.join(metadata.get('classes', label_encoder.classes_.tolist()))}\n"
                f"- Saved at: {metadata.get('saved_at', 'unknown')}"
            )

        audio_path = None
        temp_file_path = None
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
            audio_path = temp_file_path
        elif selected_sample and selected_sample != "-- choose a sample --":
            audio_path = selected_sample

        if predict_clicked and audio_path:
            try:
                result = predict_audio(audio_path, model, scaler, label_encoder)
                display_result(result, label_encoder)
                st.caption(f"Analyzed file: {Path(audio_path).name}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
        elif predict_clicked:
            st.warning("Please upload a WAV file or pick a sample before analyzing.")

    with tab_dashboard:
        st.markdown("### Evaluation dashboard")
        render_evaluation_dashboard(metadata)
        st.markdown("### Inference recipe")
        st.code(
            "Load audio -> extract MFCC-40 -> apply saved scaler -> SVM classification -> read class confidence",
            language="text",
        )

    with tab_about:
        st.markdown("### Project summary")
        about_left, about_right = st.columns([1.2, 1])
        with about_left:
            st.write(
                "This app wraps the trained depression-detection pipeline from your notebook into a clean interface. "
                "It is designed for quick demonstrations, class presentations, and local testing with custom audio."
            )
            st.write(
                "The saved artifacts are loaded directly from `best_model.pkl`, `scaler.pkl`, and `label_encoder.pkl`, "
                "so the app follows the exact prediction flow used in training."
            )
            st.warning("This is a research/demo tool, not a clinical diagnosis system.")
        with about_right:
            st.markdown("#### Key details")
            st.markdown(
                f"- Student: {metadata.get('student', 'N/A')}\n"
                f"- Registration No: {metadata.get('reg_no', 'N/A')}\n"
                f"- Train samples: {metadata.get('n_train_samples', 'N/A')}\n"
                f"- Test samples: {metadata.get('n_test_samples', 'N/A')}\n"
                f"- Overfit gap: {metadata.get('overfit_gap', 'N/A')}"
            )

    with tab_workflow:
        st.markdown("### Prediction pipeline")
        st.code(
            "Audio file -> librosa.load() -> MFCC-40 mean vector -> scaler.transform() -> SVM prediction -> confidence scores",
            language="text",
        )
        st.markdown(
            "The app uses the same feature shape as training: 40 MFCC coefficients averaged over time. "
            "If you replace the model later, keep the preprocessing aligned with the new training pipeline."
        )
        st.markdown("### Quick usage")
        st.markdown(
            "1. Upload your `.wav` file or choose a sample from the dataset.\n"
            "2. Click **Analyze Audio**.\n"
            "3. Review the predicted label, confidence, and waveform.\n"
            "4. Use the Evaluation tab to present project results.")


if __name__ == "__main__":
    main()