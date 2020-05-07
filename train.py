# USAGE
# python train.py --dataset path/to/dataset

# import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import xml.etree.ElementTree as ET 
import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from model import Detector

# function to generate dictionary of bounding boxes
# according to the labels "bad" or "good"
def findFaces(filePath):
	tree = ET.parse(filePath)
	root = tree.getroot() 
	boxes = {}

	for item in root.findall('object'):
		for child in item:
			if child.tag == "name":
				nameTag = child.text

			if child.tag == "bndbox":
				c = []
				for coords in child:
					c.append(int(coords.text))
				if nameTag not in boxes:
					boxes[nameTag] = []
				boxes[nameTag].append(c)

	return boxes

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the input dataset")
args = vars(ap.parse_args())

# initialize the parameters for the model
INIT_LR = 0.001
EPOCHS = 20
BS = 64
inputShape = (224, 224, 3)

# from the given dataset directory,
# get paths for labels and images directory
datasetDir = args["dataset"]
labelsDir = datasetDir + "/labels/"
imagesDir = datasetDir + "/images/"

# grab all labels and images path
xmlFiles = glob.glob(labelsDir + "*.xml")
imagePaths = list(paths.list_images(imagesDir))

# initialize data and labels list
data = []
labels = []

# loop over the images in imagePaths
print("[INFO] loading dataset...")
for imagePath in imagePaths:
	imgFileName = imagePath.split(os.path.sep)[-1].split(".")[0]

	# get the corresponding xml file for the grabbed image
	for xmlFile in xmlFiles:
		xmlFileName = xmlFile.split(os.path.sep)[-1].split(".")[0]
		if imgFileName == xmlFileName:
			xmlPath = xmlFile
			break

	# read the image and get the boxes co-ordinates from
	# the xml file
	image = cv2.imread(imagePath)
	boxes = findFaces(xmlPath)

	# loop over all keys in boxes
	for label, value in boxes.items():
		# loop over all the boxes for corresponding key value
		for box in value:
			try:
				# get the co-ords of the bounding box
				# crop the image and preprocess it
				(x1, y1, x2, y2) = box
				crop = image[y1:y2, x1:x2]
				crop = cv2.resize(crop, (224, 224))
				crop = img_to_array(crop)
				crop = preprocess_input(crop)

			except:
				continue

			# if desired label then add the image and 
			# the label to the data and labels list
			if label == "bad" or label == "good":
				data.append(crop)
				labels.append(label)


# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits 
# train: 80% and test: 20%
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# build the model 
print("[INFO] building model...")
model = Detector.build(inputShape=inputShape)

# compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the model
print("[INFO] training model...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# evaluate the model
print("[INFO] evaluating model...")
predIdxs = model.predict(testX, batch_size=BS)

# print the classification report 
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
