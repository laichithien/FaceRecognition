from face_recognition.face_recognition import FaceRecognition
# from face_recognition.utils import Utils
import cv2
FR = FaceRecognition()
name = 'Hop'
img = cv2.imread('./test_folder/Hop.jpg')
bboxes = FR.detect(img)
result, cropped = FR.add_face(img, bboxes, name)
for face in cropped:
    # print(face)
    cv2.imshow('Face detected', face)
    cv2.waitKey(0)
cv2.destroyAllWindows()