from keras.applications.inception_v3 import InceptionV3
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-nfl", "--no_fixed_layers", type=int , help="number of non-trainable layers")
args = vars(ap.parse_args())

# available models
MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"resnet": ResNet50
}

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
	input_shape = (299, 299)
	preprocess = preprocess_input

### train directory path
train_path = ?
generator = ImageDataGenerator(preprocessing_function = preprocess, rotation_range=360., rescale=1./255)
train_generator = generator.flow_from_directory(train_path,target_size = input_shape, shuffle = True)

#############################################################
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
base_model = Network(weights="imagenet", include_top=False)

# add a global spatial average pooling layer
x = base_model.output
print("original output" + str(x.shape))
x = GlobalAveragePooling2D()(x)
print("pooled output" + str(x.shape))

# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)

# and a logistic layer -- we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')
model.fit_generator(train_generator, steps_per_epoch=400, epochs=3)

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning all layers
model.fit_generator(train_generator, steps_per_epoch=400, epochs=20)

model.save(args['model'] + 'mdeol_name.h5')
