from ultralytics import YOLO
import os

# Use Yolo to auto-label
model = YOLO("yolov8n.pt")  # modèle pré-entraîné
model.info()

frames_dir = "data/frames"
labels_dir = "data/labels"

for img_file in os.listdir(frames_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(frames_dir, img_file)
        results = model.predict(img_path, save=False)  # inference
        with open(os.path.join(labels_dir, img_file.replace(".jpg", ".txt")), "w") as f:
            for box, cls in zip(results[0].boxes.xywhn, results[0].boxes.cls):
                x, y, w, h = box.tolist()
                f.write(f"{int(cls)} {x} {y} {w} {h}\n")
