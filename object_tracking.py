import cv2
import numpy as np

from object_detection import ObjectDetection

 

def main():


    object_detect = ObjectDetection()

    cap = cv2.VideoCapture("los_angeles.mp4")

    count = 0

    center_point_dict = dict()
    center_points_list = list()


    while True:
    
        ret, frame = cap.read()

        count += 1

        if not ret:
            break


        (class_ids, scores, boxes) = object_detect.detect(frame=frame)

        for box in boxes:
            (x, y, w, h) = box
            central_point_x = int((x + x + w) / 2)
            central_point_y = int((y + y + h) / 2)
            center_points_list.append((central_point_x, central_point_y))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for point in center_points_list:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    main()

