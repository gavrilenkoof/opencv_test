import cv2
from object_detaction_v1 import ObjectDetection


def main():
    cap = cv2.VideoCapture('los_angeles.mp4')
    

    model = ObjectDetection()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = model.detect(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()




if __name__=="__main__":
    main()