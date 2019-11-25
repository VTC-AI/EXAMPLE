#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 4000
nb_validation_samples = 1600
epochs = 50
batch_size = 16
cell_list = ("dogs","cats","chicken")
output_n = len(cell_list)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#sử dụng đó là dropout layer ,có công dụng giảm nguy cơ overfitting cho mỗi layer, ta đặt dropout sau mỗi tổ hợp conv2d và pooling, hoặc dense
model.add(Dropout(0.5))
model.add(Dense(output_n))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', # or binary_crossentropy
              optimizer='rmsprop',# or adagrad
              metrics=['accuracy'])

#chuyển ảnh thành array (tensor), bằng hàm flow_images_from_directory, thứ bậc của quy trình này bao gồm:
# nhận diện địa chỉ (folder chứa ảnh), tải toàn bộ ảnh trong folder đó, thu nhỏ kích thước còn 150x150, chuyển kết quả thành 2D Tensor,chuyển thang đo của mỗi kênh từ 0:255 thành 0:1; dán label (classes); tạo batch với kích thước 30
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=cell_list,
    seed=123
)

print(train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=cell_list,
    seed=123
)

H = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

fig = plt.figure()
plt.plot(np.arange(0, epochs), H.history['loss'], label='training loss')
plt.plot(np.arange(0, epochs), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, epochs), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, epochs), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.show()
model.save('model.h5')
