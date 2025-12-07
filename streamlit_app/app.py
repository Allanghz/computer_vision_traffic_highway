import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# Charger ton modèle entraîné
model = YOLO("runs/detect/train5/weights/best.pt")

st.title("Détection de véhicules - YOLOv8")
st.write("Voiture • Moto • Bus • Camion")

uploaded_file = st.file_uploader("Choisis une image ou une vidéo", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Sauvegarde temporaire
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Si c'est une image
    if uploaded_file.type.startswith("image"):
        results = model.predict(temp_path, save=False)
        res_plot = results[0].plot()

        st.image(res_plot, caption="Résultat YOLO")

    # Si c'est une vidéo
    else:
        st.video(uploaded_file)

        output_path = os.path.join("runs/detect/streamlit_output", uploaded_file.name)
        results = model.predict(temp_path, save=True, project="runs/detect", name="streamlit_output")

        st.success("Vidéo traitée. Résultat ci-dessous")
        st.video(output_path)
