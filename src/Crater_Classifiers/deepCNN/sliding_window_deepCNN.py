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
from keras.models import load_model

import cv2 as cv
from helper import sliding_window
import os
import csv
import numpy as np
import argparse

cwd = os.getcwd()

ap = argparse.ArgumentParser()
ap.add_argument("-tileimg", "--tileimg", type=str, default="tile3_25", help="The name of tile image")
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

print("[INFO] loading {}...".format(args["path"]))
model = load_model(args["path"])
# this script will save the sliding windows of input image and rescale it to 
# the Googlenet input size (299, 299)
# later we use these images and feed it into GoogleNet

img_name = args["tileimg"]
path = os.path.join('../data', 'images')
img = cv.imread(os.path.join(path, img_name +'.pgm'), 1)
#img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)/255.0

crater_list = []
win_sizes = range(15, 400, 5)
# loop over the image pyramid
for winS in win_sizes:
	print("Resized shape: %d, Window size: %d" % (resized.shape[0], winS))

	# loop over the sliding window for each layer of the pyramid
	# this process takes about 7 hours. To do quick test, we may try stepSize
	# to be large (60) and see if code runs OK
	#for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winS, winS)):
	for (x, y, window) in sliding_window(resized, stepSize=2, windowSize=(winS, winS)):
		# since we do not have a classifier, we'll just draw the window
		crop_img =cv.resize(window, input_shape)
		#crop_img = cv.cvtColor(crop_img,cv.COLOR_GRAY2RGB)
		img_arr = image.img_to_array(crop_img)
		img_arr = np.expand_dims(img_arr, axis=0)
		img_arr = preprocess_input(img_arr)
		preds = model.predict(img_arr)
		classes = preds.argmax()
		
		if classes == 1:
			x_c = int((x + 0.5 * winS))
			y_c = int((y + 0.5 * winS))
			crater_diameter = int(winS)
			
			crater_data = [x_c, y_c, crater_diameter, preds]
			crater_list.append(crater_data)

cnn_file = open("results/inception/"+img_name+"_sw_inception.csv","w")
with cnn_file:
    writer = csv.writer(cnn_file, delimiter=',')
    writer.writerows(crater_list)
cnn_file.close()

print("NN detected ", len(crater_list), "craters")

