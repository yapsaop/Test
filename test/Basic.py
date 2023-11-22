import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mediapipe as mp

path = 'ImageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList) 

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Lật ảnh theo trục y
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (0, 255, 0), 2)
                # cv2.putText(img, f'Face', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

                # Chuyển đổi bbox thành dạng (top, right, bottom, left) để sử dụng cho face recognition
                faceLoc = (bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0])
                encodeFace = face_recognition.face_encodings(img, [faceLoc])[0]

                # So sánh khuôn mặt nhận dạng với danh sách khuôn mặt đã biết
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    # cv2.rectangle(img, (bbox[0], bbox[3] - 35), (bbox[2], bbox[3]), (0, 255, 0), cv2.FILLED)
                    # cv2.putText(img, name, (bbox[0] + 6, bbox[3] - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 555, 555), 2)
                    cv2.putText(img, name, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                    markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Khi thoát khỏi vòng lặp, hủy bỏ tất cả cửa sổ
    cap.release()
    cv2.destroyAllWindows()

