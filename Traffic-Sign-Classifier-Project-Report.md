## Project: Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Build a Traffic Sign Recognition Project**
The goals / steps of this project are the following:
* The model should classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/Histogram.png "Dataset Histogram"
[image3]: ./examples/brightness.png "Brightness augmentation"
[image4]: ./examples/grayscale.png "Grayscale"
[image5]: ./examples/normalization.png "Normalization"
[image6]: ./examples/accuracy.png "Accuracy Plot Graph"
[image7]: ./examples/Guess.png "Top Guesses"
[image8]: ./examples/extra_images.png "Extra Samples"
[image9]: ./examples/FeatureMap_img.png "Feature Map image sample"
[image10]: ./examples/FeatureMap_conv1.png "First Convolutional Layer"
[image11]: ./examples/FeatureMap_conv2.png "Second Convolutional Layer5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the pandas library specificly the pd.DataFrame() function to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the 3 datasets showing the frequency of each traffic sign in each dataset.

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to increase the brightness of the images by converting the image to HSV the scaling up the V channel and converting it back to RGB. This was adapted from a post from Vivek Yadav on Medium. 

Here is an example of a traffic sign image after Brightness Augmentation.

![alt text][image3]

Then I converted it to grayscale since the traffic signs have different shapes which could be enough to classify the sign.

![alt text][image4]

As a last step, I normalized the image data to have more reliable neural network as for each feature to have a similar range so that our gradients don't go out of control. 


#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16   |    
| RELU                  |                                               |
| Max pooling		    | 2x2 stride,  outputs 5x5x16        			|
| Flatten				| Output = 400 									|
| Fully Connected		| Output = 120 									|
| RELU					|												|
| Dropout               |                                               |
| Fully Connected	    | Output = 84                       			|
| RELU  				|            									|
| Dropout       		|            									|
| Fully Connected		| Output = 43									|
 


#### 3. Training the Model

To train the model, I used
* AdamOptimizer
* EPOCHS = 60
* BATCH_SIZE = 128
* rate = 0.0005
* keep_prob = 1 , 0.5

#### 4. Reaching the Accuracy more than 0.93

![alt text][image6]

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.955
* test set accuracy of 0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture was the lenet that was used to classify the letters.

The problem was the it's accuracy didn't exceed 0.9 so by adding the brightness augmentation and the dropout it showed much progress.

* Tuning the Learning rate and the Keep_prob of the dropout had a significant change to the accuracy.

* The dropout made the model more reliable as it is a regularization technique for reducing overfitting.

The model has an acceptable accuracy however by adding more augmented dataset it will be more accurate and reliable.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image8]

The images I picked was traffic sign street art where some stickers are added on the sign interfering some parts of the sign. Also some images are not captured from the front view.
#### 2. Prediction of new set of images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| no entry      		| no entry   									| 
| Predestrian     		| Predestrian 									|
| no entry				| no entry										|
| Stop Sign	      		| Stop Sign                                     |
| go left   		    | go left                                       |


The model was able to correctly guess extra traffic signs successfully

#### 3. The top 5 softmax probabilities for 6 new images along.

![alt text][image7]



### Visualizing the Neural Network
![alt text][image9]
#### First Convolutional Layer
The Feature Map shows that the Neural network looked into the sides of the triangular shape

![alt text][image10]

#### Second Convolutional Layer
![alt text][image11]
