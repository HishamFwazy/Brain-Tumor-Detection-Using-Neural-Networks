import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
#print(os.listdir(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset"))


os.listdir("../")


# import shutil
# from tqdm import tqdm
#
#
# for i in tqdm(os.listdir(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set")):
# #     print(i)
#     if i.split(".")[0] == "no":
#         shutil.copy2(os.path.join(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set",i),os.path.join(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set\no",i))
#     elif i.split(".")[0] == "yes":
#         shutil.copy2(os.path.join(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set",i),os.path.join(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set\yes",i))
#


import keras
from keras.models import Model, load_model
from keras.layers import Dense
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


#os.listdir(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\training_set")

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=r"E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\augmented data",target_size=(224,224))


tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=r"E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\augmented data", target_size=(224,224))



from keras.applications.vgg16 import VGG16

vggmodel = VGG16(weights='imagenet', include_top=True)

vggmodel.summary()


for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False



X= vggmodel.layers[-2].output


predictions = Dense(2, activation="softmax")(X)


model_final = Model(inputs = vggmodel.input, outputs = predictions)


model_final.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])


model_final.summary()



from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

os.listdir("../")

filepath="cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}"
#save the model with the best validation (development) accuracy till now
checkpoint = ModelCheckpoint(r"E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\Modelss/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))


#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')

#hist = model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 10, validation_data= testdata, validation_steps=1, callbacks=[checkpoint])

#
best_model = load_model(filepath=r'E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\Modelss\cnn-parameters-improvement-27-0.84.model')
best_model.metrics_names
#model_final.save_weights("vgg16_1.h5")


# import pandas as pd
# df=pd.read_csv(r"D:\2NAAAAA\gradution project\New folder (2)\brain_tumor_dataset\single_prediction\Y33.jpg")

# print(df["label"][0])
# pd.options.mode.chained_assignment = None  # default='warn'

img = image.load_img(os.path.join(r"E:\SSD\Uni\Graduation Project\Brain-Tumor-Detection-master\augmented data\no\aug_N1_0_328.jpg"),target_size=(224,224))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)
output = best_model.predict(img)
if output[0][0] > output[0][1]:
 print("no")
else:
 print('yes')