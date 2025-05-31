import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == "__main__":
    model = RTDETR("my_cfg/rtdetr-r18.yaml").load("weights/rtdetr-r18.pt")

    model.train(
        data="dataset/A_drowning_person.yaml",
        cache=True,
        imgsz=640,
        epochs=100,
        batch=4,
        workers=4,
        device="0",
        # resume="runs\\train\\test\\rtdetr-mobilenetv4\\weights\\last.pt",  # last.pt path
        optimizer="AdamW",
        project="runs/train",
        name="test/1",
    )
