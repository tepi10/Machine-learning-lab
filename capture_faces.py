import cv2
import os

# person's name
person_name = "meheti"

dataset_path = "dataset/" + person_name

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face,(100,100))

        file_name = dataset_path + "/" + str(count) + ".jpg"

        cv2.imwrite(file_name, face)

        count += 1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Capturing Faces",frame)

    if cv2.waitKey(1)==27 or count>=100:
        break

cap.release()
cv2.destroyAllWindows()