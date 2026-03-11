import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

dataset_path = "dataset"

faces = []
labels = []

label_map = {}
label = 0

for person in os.listdir(dataset_path):

    label_map[label] = person

    person_path = os.path.join(dataset_path,person)

    for img in os.listdir(person_path):

        img_path = os.path.join(person_path,img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image,(100,100))

        faces.append(image)
        labels.append(label)

    label += 1

faces = np.array(faces)
labels = np.array(labels)

faces = faces/255.0

faces = faces.reshape(len(faces),100*100)

labels = to_categorical(labels)

X_train,X_test,y_train,y_test = train_test_split(
    faces,labels,test_size=0.2
)

model = Sequential()

model.add(Dense(512,activation='relu',input_shape=(10000,)))

model.add(Dense(256,activation='relu'))

model.add(Dense(labels.shape[1],activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test,y_test)
)

model.save("face_model.h5")

np.save("labels.npy",label_map)

print("Training Completed")