import torch
import numpy as np


class ObjectDetection:

    MODELS = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

    def __init__(self, model_name='yolov5s'):
        print("Running pytorch YOLOv5")

        self.check_model_from_available(model_name)

        torch.hub.set_dir('models')

        try:
            path = 'models/ultralytics_yolov5_master/'
            self.model = torch.hub.load(path, model=model_name, source='local', device='cuda:0')
        except FileNotFoundError:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, device='cuda:0')

        device = torch.device(0)
        self.model.to(device)

    
    def detect(self, frame, size=640):
        results = self.model(frame, size)
        frame = np.array(results.render()[0]) 

        return frame

    def check_model_from_available(self, model_name):
        if model_name not in ObjectDetection.MODELS:
            print(f"Not found {model_name}. Use default model: yolov5s")
            model_name = "yolov5s"