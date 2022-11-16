import cv2
import numpy as np

from object_detection import ObjectDetection

import math

 

def main():


    object_detect = ObjectDetection()

    cap = cv2.VideoCapture("los_angeles.mp4")

    count = 0

    center_point_dict = dict()
    # center_points_list = list()
    center_point_prv_frame = list()

    tracking_objects = dict()
    track_id = 0

    while True:
    
        ret, frame = cap.read()

        count += 1

        if not ret:
            break

        center_point_current_frame = list()


        (class_ids, scores, boxes) = object_detect.detect(frame=frame)

        for box in boxes:
            (x, y, w, h) = box
            central_point_x = int((x + x + w) / 2)
            central_point_y = int((y + y + h) / 2)
            center_point_current_frame.append((central_point_x, central_point_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for point in center_point_current_frame:
            for point2 in center_point_prv_frame:
                distance = math.hypot(point2[0] - point[0], point2[1] - point[1])

                if distance < 10:
                    tracking_objects[track_id] = point
                    track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        center_point_prv_frame = center_point_current_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()

