from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26l-seg.pt")

    results = model.train(data="tomato4yolo/dataset.yaml", epochs=100, imgsz=480, batch=0.6)
