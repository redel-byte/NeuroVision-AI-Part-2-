import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

# MLflow normally retries for minutes; keep this interactive app responsive.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

from neurovision_ai_2.inference import CLASSES, predict_image


st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="centered")

st.title("NeuroVision AI")
st.write("Upload one brain MRI image to see the model's prediction.")
st.warning("Outil pédagogique d’aide à la décision — pas un diagnostic médical.")

model_choice = st.radio(
    "Model source",
    ["Offline checkpoint demo", "MLflow champion"],
    horizontal=True,
)
source = "local" if model_choice == "Offline checkpoint demo" else "champion"
if source == "local":
    st.info("Offline demo: this local checkpoint is not verified as the MLflow champion.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png", "bmp"],  # Same formats as predict_image().
    max_upload_size=20,  # Limit one upload to 20 MB.
)

if uploaded_file is None:
    st.info("Choose an image to begin.")
else:
    try:
        st.image(uploaded_file, caption="Uploaded image")  # Preview before prediction.
    except Exception:
        st.error("This file is not a readable image. Choose another MRI image.")
    else:
        if st.button("Analyze image", type="primary"):
            try:
                # The uploader gives us bytes, but predict_image() needs a file path.
                suffix = Path(uploaded_file.name).suffix.lower()
                with TemporaryDirectory() as temp_dir:  # Removes the image afterward.
                    image_path = Path(temp_dir) / f"uploaded_image{suffix}"
                    image_path.write_bytes(uploaded_file.getvalue())

                    with st.spinner("Loading model and analyzing image..."):
                        result = predict_image(image_path, source=source)
            except ConnectionError as error:
                st.error(str(error))  # Usually means the MLflow server or SSH tunnel is down.
            except FileNotFoundError as error:
                st.error(str(error))
            except (OSError, ValueError) as error:
                st.error(f"Could not read this image: {error}")
            except Exception as error:
                if source == "local":
                    st.error(f"Local checkpoint could not run: {error}")
                else:
                    st.error(f"MLflow champion could not run: {error}")
            else:
                st.subheader("Model result")
                st.metric("Predicted class", result["label"].capitalize())
                st.write(f"Top model score: {result['confidence']:.1%}")
                if result["model_source"] == "local":
                    st.caption("Source: local checkpoint, not the MLflow champion.")
                else:
                    st.caption(f"MLflow champion model, version {result['model_version']}")

                st.subheader("Scores for all four classes")
                for name in CLASSES:  # Keep the same class order used in training.
                    score = float(result["probabilities"][name])
                    st.write(f"{name.capitalize()}: {score:.1%}")
                    st.progress(score)

                st.caption("Model scores are not calibrated medical probabilities.")
