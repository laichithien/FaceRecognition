from face_recognition.face_recognition import FaceRecognition
import cv2
import time

FR = FaceRecognition()
vid = cv2.VideoCapture(1)
cnt = 0 
while True:
    ret, frame = vid.read()
    # print(frame)
    if ret is None:
        continue
    if frame is None:
        continue
    # print(frame.shape)
    t1 = time.time()
    bboxes = FR.detect(frame)
    
    result = FR.recognize(frame, bboxes)
    t2 = time.time()
    fps = 1/(t2-t1)
    print(f'fps: {fps}')
    if result is None:
        cv2.imshow('Face Recognition', frame)
        continue
    img_result = FR.draw_detect_result(frame, bboxes, result)
    cv2.imshow('Face Recognition', img_result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()


