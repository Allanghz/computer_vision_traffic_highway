from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # modèle pré-entraîné
results = model.train(data="data.yaml", epochs=50, imgsz=640)