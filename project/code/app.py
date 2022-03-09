# %%
# Based on https://www.tensorflow.org/tutorials/images/transfer_learning

# %%
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
PATH = "../input/"

train_path = os.path.join(PATH, 'train/')
test_path = os.path.join(PATH, 'test/')

train_df = pd.read_csv(PATH + 'train.csv')
test_df = pd.read_csv(PATH + 'test.csv')

train_jpg = glob(PATH + "train/*.jpg")
test_jpg = glob(PATH + "test/*.jpg")

# %%
#Modify the Id such that each Id is the full image path. In the form
def train_id_to_path(x):
    return train_path + x + ".jpg"
def test_id_to_path(x):
    return test_path + x + ".jpg"

#Read in the data and drop unnecessary columns
train = train_df.drop(['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'],axis=1)
test = test_df.drop(['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'],axis=1)

#Add the .jpg extensions to the image file name ids
train["img_path"] = train["Id"].apply(train_id_to_path)
test["img_path"] = test["Id"].apply(test_id_to_path)

# %%
#Set the size image you want to use
image_height = 128
image_width = 128

#define a function that accepts an image url and outputs an eager tensor
def path_to_eagertensor(image_path):
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (image_height, image_width))
    return image


# %%
#get all the images in the training folder and put their tensors in a list
X = []
for img in train['img_path']:
    new_img_tensor = path_to_eagertensor(img)
    X.append(new_img_tensor)

X = np.array(X)

# %%
#get all the images in the test folder and put their tensors in a list
X_submission = []
for img in test['img_path']:
    new_img_tensor = path_to_eagertensor(img)
    X_submission.append(new_img_tensor)
    
print(type(X_submission),len(X_submission))
X_submission = np.array(X_submission)
print(type(X_submission),X_submission.shape)

# %%
y = train['Pawpularity']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# %%
# Create the base model from the pre-trained model EfficientNet B0
base_model = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=(128,128,3),
                pooling=None,
                classifier_activation=None
            )

# %%
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

# %%
# Building model
inputs = tf.keras.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# %%
# Compiling model using mse ass loss function and Adam as optimizer
base_learning_rate = 0.0001
model.compile(optimizer='Adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])

# %%
# Building tool for data augmentation
data_augmentation = ImageDataGenerator(rotation_range=15, zoom_range=0.15, width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.1,
    horizontal_flip = True, 
    fill_mode = "nearest")

# %%
# Setting early stopping as callback
kall = tf.keras.callbacks.EarlyStopping(monitor='val_rmse',patience=10,restore_best_weights=True)


# %%
# Training the model
history = model.fit(
    data_augmentation.flow(x_train,y_train,batch_size=300),
    validation_data = (x_test,y_test),
    steps_per_epoch = len(x_train) // 300,
    epochs = 60, callbacks = [kall]
)

# %%
#predict on the submission data
cnn_pred = model.predict(X_submission)
print(X_submission.shape, type(X_submission))
print(cnn_pred.shape, type(cnn_pred))

# %%
# Building csv file with the outputs
cnn = pd.DataFrame()
cnn['Id'] = test['Id']
cnn['Pawpularity'] = cnn_pred
cnn.to_csv('../output/submission.csv',index=False)


