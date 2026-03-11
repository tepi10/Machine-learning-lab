import cv2
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("face_model.h5")

label_map = np.load("labels.npy",allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h,x:x+w]

        face = cv2.resize(face,(100,100))

        face = face/255.0

        face = face.reshape(1,10000)

        prediction = model.predict(face)

        label = np.argmax(prediction)

        name = label_map[label]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,name,(x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()