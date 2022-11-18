import torch
import numpy as np


class ObjectDetection:
    def __init__(self):
        print("Running pytorch YOLOv5")

        # try:

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', device='cuda:0')
        # except:
            # print("Download from local rep")
            # path = 'yolov5'
            # self.model = torch.hub.load(path, model='yolov5x', source='local')
        device = torch.device(0)
        self.model.to(device)

    
    def detect(self, frame, size=640):
        results = self.model(frame, size)
        frame = np.array(results.render()[0]) 

        return frame
