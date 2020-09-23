# Hand Gesture Recognition (Perceptron and CNN)

Goal is to recognize hand gestures. I've trained the model on my own dataset using *Perceptron* (and *CNN* thereafter). I've included my dataset in the repository itself. In it's present state the model is trained to recognize just two (now four) gestures (please see the *CNN* implementation section in this readme, I trained it for 4 gestures afterwards) but can easily be trained for many hand gestures.
I'll also upload the code that I'm using for capturing the hand an processing it for training the model.

Model gives a high testing accuracy of about **99%** using just two *Dense* layers. But if you want to train more hand gestures then you'll probably need more a complex network.

Images in the dataset are of dimension *200 by 200*. But for performance reasons they have been resized to *50 by 50*. You can use them as it is if you have a powerful setup otherwise program displays a Tensorflow error 
`tensorflow/core/framework/allocator.cc:101] Allocation of X exceeds 10% of system memory.`

## What's in the Repository

* `captureHand.py` - This program can capture new hand gestures and write them in the specified directory
* `recognizer.py` - This is the main program that uses pretrained model (in the repo) for recognizing hand gestures
* `trainer.py` - This program uses the given dataset to train the Perceptron model
* `modelWeights.h5` - Weights for the Perceptron model
* `trainedModel.json` - JSON format of the model
* `CNN Model` - A directory that contains CNN model implementation for the same recognition purpose (with 4 gestures)

## Sample of images in the Dataset

* First Hand Gesture

![firstHandGesture.jpg](firstHandGesture.jpg)

* Second Hand Gesture

![secondHandGesture.jpg](secondHandGesture.jpg)

## Required External libraries

* `cv2 (opencv)`
* `imutils`
* `glob`
* `sklearn (scikit-learn)`
* `keras`
* `numpy`

## What You Should See

* For the first gesture

![output1.jpg](output1.jpg)

* For the second gesture

![output2.jpg](output2.jpg)

## Future

I hope to implement more than two gestures in the future. There will be further improvements in the code itself too.

# CNN Implementation

As of May 1, 2019 I have used CNN for the recognition of 4 gestures. All the required files are present in the folder `CNN Model`.

### Sample of Images in the New Dataset

![gesture0.jpg](CNN%20Model/gesture0.jpg)

![gesture1.jpg](CNN%20Model/gesture1.jpg)

![gesture2.jpg](CNN%20Model/gesture2.jpg)

![gesture5.jpg](CNN%20Model/gesture5.jpg)

## Outputs Showing Error %ages

* *CNN* Training Output

![CNNoutput.jpg](CNN%20Model/outputCNN.JPG)

* *Perceptron* Training Output

![PERCEPTRONoutput.jpg](CNN%20Model/outputPERCEPTRON.JPG)

## Improvements and Differences

I've used both the *Perceptron* as well as the *CNN* Model for recognition of 4 hand gestures. And I can easily say *CNN* works better for extracting features of an image. There's and improvement of Error %age from **32%** to **5%**. This happens because Perceptron model is a simple 1 dimensional set of neurons, therefore it reduces a lot of features in the images whereas the *CNN* is specially designed to work with images and works on 2 dimensional set of neurons. You can read more about the algorithms online.
