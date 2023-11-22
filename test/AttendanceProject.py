import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

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

cap = cv2.VideoCapture(0)
f_01 = np.load('./face_01.npy').reshape(20, 50*50*3)
f_02 = np.load('./face_02.npy').reshape(20, 50*50*3)

while True:
    success, img = cap.read()
    face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faces = face_cas.detectMultiScale(imgS, 1.3, 5) #

    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print (faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,555,555),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Khi thoát khỏi vòng lặp, hủy bỏ tất cả cửa sổ
cv2.destroyAllWindows()

    

