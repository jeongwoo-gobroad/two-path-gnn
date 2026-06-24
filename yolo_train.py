from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo26l-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="tomato4yolo/dataset.yaml", epochs=100, imgsz=480, batch=0.6)