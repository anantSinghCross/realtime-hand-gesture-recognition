# CODE THAT USES THE GIVEN DATASET TO TRAIN THE CNN MODEL

import numpy
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import glob
import cv2
from sklearn.utils import shuffle
import os

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
imageSize = 50

# load training data for gesture 2
myFiveTrainImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/fiveFingerTrainDataset/*.jpg")
myFiveTrainImageFiles.sort()
myFiveTrainImages = [cv2.imread(img,0) for img in myFiveTrainImageFiles] #we pass zero to load greyscale image

for i in range(0,len(myFiveTrainImages)):
    myFiveTrainImages[i] = cv2.resize(myFiveTrainImages[i],(imageSize,imageSize))
tn1 = numpy.asarray(myFiveTrainImages)

# load training data for gesture 1
myZeroTrainImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/zeroFingerTrainDataset/*.jpg")
myZeroTrainImageFiles.sort()
myZeroTrainImages = [cv2.imread(img,0) for img in myZeroTrainImageFiles]

for i in range(0,len(myZeroTrainImages)):
    myZeroTrainImages[i] = cv2.resize(myZeroTrainImages[i],(imageSize,imageSize))
tn2 = numpy.asarray(myZeroTrainImages)

# load training data for gesture 3
myGes3TrainImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/gesture3Train/*.jpg")
myGes3TrainImageFiles.sort()
myGes3TrainImages = [cv2.imread(img,0) for img in myGes3TrainImageFiles]

for i in range(0,len(myGes3TrainImages)):
    myGes3TrainImages[i]= cv2.resize(myGes3TrainImages[i],(imageSize,imageSize))
tn3 = numpy.asarray(myGes3TrainImages)

# load testing data for gesture 4 (RANDOM)
myGes4TrainImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/gesture4Train/*.jpg")
myGes4TrainImageFiles.sort()
myGes4TrainImages = [cv2.imread(img,0) for img in myGes4TrainImageFiles]

for i in range(0,len(myGes4TrainImages)):
    myGes4TrainImages[i]= cv2.resize(myGes4TrainImages[i],(imageSize,imageSize))
tn4 = numpy.asarray(myGes4TrainImages)

finalTrainImages = []
finalTrainImages.extend(myFiveTrainImages)
finalTrainImages.extend(myZeroTrainImages)
finalTrainImages.extend(myGes3TrainImages)
finalTrainImages.extend(myGes4TrainImages)


# load testing data for gesture 2 (STOP)
myFiveTestImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/fiveFingerTestDataset/*.jpg")
myFiveTestImageFiles.sort()
myFiveTestImages = [cv2.imread(img,0) for img in myFiveTestImageFiles]

for i in range(0,len(myFiveTestImages)):
    myFiveTestImages[i] = cv2.resize(myFiveTestImages[i],(imageSize,imageSize))
ts1 = numpy.asarray(myFiveTestImages)

# load testing data for gesture 1 (FIST)
myZeroTestImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/zeroFingerTestDataset/*.jpg")
myZeroTestImageFiles .sort()
myZeroTestImages = [cv2.imread(img,0) for img in myZeroTestImageFiles]

for i in range(0,len(myZeroTestImages)):
    myZeroTestImages[i] = cv2.resize(myZeroTestImages[i],(imageSize,imageSize))
ts2 = numpy.asarray(myZeroTestImages)

# load testing data for gesture 3 (OK)
myGes3TestImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/gesture3Test/*.jpg")
myGes3TestImageFiles.sort()
myGes3TestImages = [cv2.imread(img,0) for img in myGes3TestImageFiles]

for i in range(0,len(myGes3TestImages)):
    myGes3TestImages[i]= cv2.resize(myGes3TestImages[i],(imageSize,imageSize))
ts3 = numpy.asarray(myGes3TestImages)

# load testing data for gesture 4 (RANDOM)
myGes4TestImageFiles = glob.glob("C:/Users/anant singh/Desktop/Dataset/gesture4Test/*.jpg")
myGes4TestImageFiles.sort()
myGes4TestImages = [cv2.imread(img,0) for img in myGes4TestImageFiles]

for i in range(0,len(myGes4TestImages)):
    myGes4TestImages[i]= cv2.resize(myGes4TestImages[i],(imageSize,imageSize))
ts4 = numpy.asarray(myGes4TestImages)


finalTestImages = []
finalTestImages.extend(myFiveTestImages)
finalTestImages.extend(myZeroTestImages)
finalTestImages.extend(myGes3TestImages)
finalTestImages.extend(myGes4TestImages)


x_train = numpy.asarray(finalTrainImages)
x_test = numpy.asarray(finalTestImages)

# Now preparing the training and testing outputs

y_myFiveTrainImages = numpy.empty([tn1.shape[0]])
y_myZeroTrainImages = numpy.empty([tn2.shape[0]])
y_myGes3TrainImages = numpy.empty([tn3.shape[0]])
y_myGes4TrainImages = numpy.empty([tn4.shape[0]])


y_myFiveTestImages = numpy.empty([ts1.shape[0]])
y_myZeroTestImages = numpy.empty([ts2.shape[0]])
y_myGes3TestImages = numpy.empty([ts3.shape[0]])
y_myGes4TestImages = numpy.empty([ts4.shape[0]])


for j in range(0,tn1.shape[0]):
    y_myFiveTrainImages[j] = 5

for j in range(0,tn2.shape[0]):
    y_myZeroTrainImages[j] = 0

for j in range(0,tn3.shape[0]):
    y_myGes3TrainImages[j] = 1

for j in range(0,tn4.shape[0]):
    y_myGes4TrainImages[j] = 2

for j in range(0,ts1.shape[0]):
    y_myFiveTestImages[j] = 5

for j in range(0,ts2.shape[0]):
    y_myZeroTestImages[j] = 0
    
for j in range(0,ts3.shape[0]):
    y_myGes3TestImages[j] = 1
    
for j in range(0,ts4.shape[0]):
    y_myGes4TestImages[j] = 2

y_train_temp = []
y_train_temp.extend(y_myFiveTrainImages)
y_train_temp.extend(y_myZeroTrainImages)
y_train_temp.extend(y_myGes3TrainImages)
y_train_temp.extend(y_myGes4TrainImages)
y_train = numpy.asarray(y_train_temp)

y_test_temp = []
y_test_temp.extend(y_myFiveTestImages)
y_test_temp.extend(y_myZeroTestImages)
y_test_temp.extend(y_myGes3TestImages)
y_test_temp.extend(y_myGes4TestImages)
y_test = numpy.asarray(y_test_temp)

print(x_train.shape)
#print(x_test.shape)

print(y_train.shape)
#print(y_test.shape)
print()
print(x_test.shape)
print(y_test.shape)
#shuffling the data
x_train,y_train = shuffle(x_train,y_train)
x_test,y_test = shuffle(x_test,y_test)

# flatten 50*50 images to a 2500 vector for each image
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], 1,imageSize,imageSize).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1,imageSize,imageSize).astype('float32')

# normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
print("num_classes")
print(num_classes)
# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, imageSize, imageSize), data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=20, verbose=1)

# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

# Save the model
model_json = model.to_json();
with open("trainedModel.json","w") as jsonFile:
    jsonFile.write(model_json)
model.save_weights("modelWeights.h5")
print("Saved model to disk")