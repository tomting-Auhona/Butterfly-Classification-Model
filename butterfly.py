import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

root_dir = "D:/butterfly/leedsbutterfly/images"

import os
import cv2

data = []
labels = []

for file in os.listdir(root_dir):
    label = int(str(file)[:3])
    img = cv2.imread(root_dir + "/" + file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)
    labels.append(label)

import numpy as np
import pandas as pd

data = np.array(data) / 255
labels = np.array(labels)
labels = labels.reshape(832, 1)
from sklearn.preprocessing import OneHotEncoder

onehot = OneHotEncoder()
labels = onehot.fit_transform(labels)
labels.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
from keras.applications import Xception
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model

base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(10, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest"
)

dataflow = datagen.flow(
    x=X_train,
    y=y_train.toarray(),
    batch_size=32
)
history = model.fit_generator(
    dataflow,
    steps_per_epoch=len(X_train) // 32,
    validation_data=(X_test, y_test.toarray()),
    validation_steps=len(X_test) // 32,
    epochs = 10
)

history_df = pd.DataFrame(history.history)
history_df.head()

import matplotlib.pyplot as plt

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "valid"])
plt.show()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "valid"])
plt.show()

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test.toarray(), axis=1)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

model.save("butterfly.h5")