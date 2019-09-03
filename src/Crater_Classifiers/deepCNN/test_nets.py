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
from keras.models import load_model

import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
ap.add_argument("-path", "--path", type=str , help="path to the model to load")
args = vars(ap.parse_args())

MODELS = {
	"vgg16": VGG16,
	"inception": InceptionV3,
	"resnet": ResNet50
}

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception"):
	input_shape = (299, 299)
	preprocess = preprocess_input

# give the test path directory
test_path = ?
generator = ImageDataGenerator(preprocessing_function = preprocess, rescale=1./255)
test_generator = generator.flow_from_directory(test_path,target_size = input_shape, shuffle = False)

print("[INFO] loading {}...".format(args["path"]))
model = load_model(args["path"])


# classify the images
print("[INFO] classifying images with '{}'...".format(args["model"]))
preds = model.predict_generator(test_generator)

label_map = (test_generator.class_indices)
classes = preds.argmax(axis=-1)

