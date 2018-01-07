# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[gif1]: ./data/video.gif "Video"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network that has been adapted from the NVIDIA end-to-end self-driving car demo. It consists of a cropping layer that excludes the hood of tha car and some of the extraneous scenery. The data is normalized in the model using a Keras lambda layer (code line 189). It also uses a combination of 5x5 and 3x3 filter sizes and depths between 24 and 64 for the 2D convolutional layers (model.py lines 192-196). The result of these layers is flattened and goes through the fully-connected layers that produce an estimate of the steering angle. Each of the fully-connected and convolutional layers employ the ReLU activation function to introduce non-linearity to the model. This model is thus a regression network. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 191). 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The car drives similarly to the collector of the test data on the test track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 205).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (5 laps), recovering from the left and right sides of the road (3 laps), and smooth turning data (3 laps). The inclusion of all of the above data was necessary to allow for the car to drive through the entire track without lunging into the water or hillside.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one defined in the NVIDIA paper. I thought this model might be appropriate because it has been shown to work with success for the same application on the paper. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I chose an 80-20 split. I found that my first model had a low mean squared error on the training set but an increasingly high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include a dropout layer with a value of 0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. This might have been due to the model underfitting the data. I then increased the number of epochs to 10 to see the behaviour of the error. The model took longer to train and it became apparent that the validation error did not improve further after approximately 7 epochs. At this point, the simulator was still not able to get around the first turn.

I thus decided to increase my training data to include more frames with only smooth turn data. I augmented the resulting training data with correction data where I captured frames where I would steer back to the center of the road from the edge of the road. This data trains the model to react to situations that could result in the car wearing off the track.  

![Alt Text][gif1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture implemented using Keras:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB images   							| 
| Lambda         		| Normalize and mean center the input image data   							| 
| Cropping         		| Crop the hood of the car and unnecessary scenery from each 2D frame   	| 
| Dropout         		| Exclude a portion of the images to avoid overfitting				| 
| Convolution      	| 5x5 kernel with a 2x2 stride 	|
| RELU					|		Rectified Linear Unit serves as the activation function andd adds non-linearity to the model										|
| Convolution      	| 5x5 kernel with a 2x2 stride 	|
| RELU					|		Rectified Linear Unit serves as the activation function andd adds non-linearity to the model										|
| Convolution      	| 5x5 kernel with a 2x2 stride 	|
| RELU					|		Rectified Linear Unit serves as the activation function andd adds non-linearity to the model										|
| Convolution      	| 5x5 kernel with a 2x2 stride 	|
| RELU					|		Rectified Linear Unit serves as the activation function andd adds non-linearity to the model										|
| Convolution      	| 5x5 kernel with a 2x2 stride 	|
| RELU					|		Rectified Linear Unit serves as the activation function andd adds non-linearity to the model										|
| Flatten | Convert the 2D output thus far into a 1D array |
| Fully connected		| 1st of 3 FC layers        									|
| RELU					|	Rectified Linear Unit serves as the activaation function and adds non-linearity to the model											|
| Fully connected		| 2nd of 3 FC layers        									|
| RELU					|		Rectified Linear Unit serves as the activation function and adds non-linearity to the model										|
| Fully connected		| 3rd of 3 FC layers that produces an estimate for the steering angle	|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would balance the left biased data. Since most of the turns were to the left. For example, here is an image that has then been flipped:

![alt text][image6]

Here is an example of cropped data. Only the portion of the image that was deemded to be useful for the training process was used. 
![alt text][image7]

After the collection process, I had 15,973 number of data points after the data acquisition process. The data augmentation steps defined above helped increase the total no. of data points to 63,892. I then preprocessed this data by normalizing it.

The file model.py also contains (commented out) code that attempts to minimize the effect of a steering angle of 0 deg. by only using a randomly sampled subset of frames that have said steering angle. This was to introduce more uniformity in the data. However, this was not implemented in the final version of the code.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by hte trend of the validatioon error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
