import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def get_bounding_box(image, threshold=1):
    mask = image < threshold
    if False not in mask:
        return image
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    return image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]


# create 32x32 patches representing each letter in the alphabet
n_rows = 32
n_cols = 32
patch_shape = (32, 32)
alphabet = 'abcdefghijlmnopqrstuvxz'
letters = []
for letter in alphabet:
    im = Image.new("RGB", patch_shape)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("fonts/giordano/giordano-webfont.ttf", 32)
    draw.text((8, 4), letter, font=font)
    data = np.array(im.convert('L'))
    data = get_bounding_box(data)
    data = resize(data, patch_shape)
    data = data.reshape((1,) + patch_shape)
    letters.append(data)
letters = np.array(letters, dtype=np.float32)
labels = np.eye(len(letters))[np.arange(len(letters))]


# initialize the CNN model
model = Sequential()
model.add(
    Conv2D(32, (3, 3), input_shape=(1, 32, 32), data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(labels)))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
batch_size = 128

# augmentation configuration for training
train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    data_format='channels_first')
# estimate std, mean, ZCA from baseline data
train_datagen.fit(letters)

# augmentation configuration for testing:
validation_datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    data_format='channels_first')

# generator will read pictures found in array
train_generator = train_datagen.flow(
        letters, labels, batch_size=batch_size)

# generator will read pictures found in array
validation_generator = validation_datagen.flow(
        letters, labels, batch_size=batch_size)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=500,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

model.save_weights('first_try.h5')
