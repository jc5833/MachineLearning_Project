import os
import time
import math
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
VEHICLES= '/content/data/vehicles'
NONVEHICLES= '/content/data/nonvehicles'

vehicleList = os.listdir(VEHICLES)
nonvehicleList = os.listdir(NONVEHICLES)


print('Number of vehicle images:', len(vehicleList))
print('Number of nonvehicle images:', len(nonvehicleList))
Base = '/tmp/'
Train = os.path.join(Base, 'train')
Test = os.path.join(Base, 'test')

Vehicle_Train = os.path.join(Train, 'vehicles')
Nonvehicle_Train = os.path.join(Train, 'non-vehicles')

Vehicle_Test = os.path.join(Test, 'vehicles')
Nonvehicle_Test = os.path.join(Test, 'non-vehicles')
train_size = .8
train_vehicles, test_vehicles = train_test_split(
    vehicleList, train_size=train_size, shuffle=True, random_state=1
)

train_non_vehicles, test_non_vehicles = train_test_split(
    nonvehicleList, train_size=train_size, shuffle=True, random_state=1
)
def move_images(image_list, old_dir_path, new_dir_path):
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    for file_name in image_list:
        shutil.copy(
            os.path.join(old_dir_path, file_name),
            os.path.join(new_dir_path, file_name)
        )
    print(f'{len(image_list)} IMAGES COPIED TO {new_dir_path}')
move_images(train_vehicles, VEHICLES, Vehicle_Train)
move_images(train_non_vehicles, NONVEHICLES, Nonvehicle_Train)

move_images(test_vehicles, VEHICLES, Vehicle_Test)
move_images(test_non_vehicles, NONVEHICLES, Nonvehicle_Test)
Img_Size= 256

train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


train_generator = train_generator.flow_from_directory(
    Train,
    target_size=(Img_Size, Img_Size),
    shuffle=True,
    class_mode='binary'
)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


test_generator = test_datagen.flow_from_directory(
    Test,
    target_size=(Img_Size, Img_Size),
    class_mode='binary'
)
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(Img_Size, Img_Size, 3)),
    layers.MaxPool2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5
)

model_path = 'vehicle_detection.h5'
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_path,
    save_best_only=True
)
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=7,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping, model_checkpoint]
)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

