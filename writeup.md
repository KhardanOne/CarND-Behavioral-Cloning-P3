# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./doc/model_architecture.jpg "Model Architecture"
[image2]: ./doc/center.jpg "Center Image"
[image3]: ./doc/preprocessed_images.jpg "Cropped and flipped images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* data_manip.py containing image manipulation functions
* reader.py containing csv and image reading functions
* cfg.py containing configurations such as directories, filenames
* this writeup.md summarizing the results


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Both tracks can be driven in both directions. In order to drive in the reverse direction use manual override mode by pressing W or S keys. Turn back, then release the controls. The car will start driving autonomously.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

All 27500 data iteams are loaded into the memory. With 32 gigabites of RAM it was no problem, but with 16 or less it might be. Loading the data and training takes ca. 1.5 minutes on an average PC (as of 2020).

I used Tensorflow 2.3.1 with Keras 2.4.3. It should work well also inside Udacity Anaconda environment according to my tests.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model starts with image cropping and normalization (model.py lines 38-39) 

They are followed by 3 convolution layers ranging from 7 * 7 to 3 * 3 kernel sizes, interspersed with max-pooling with 4 * 4 kernel sizes and RELUs. The very first convolitional layer uses 2 * 2 stride to decrease the size (model.py lines 40-48).

After the flattening step the resulting width of the fully connected layer is 1280. After a Dropout layer of 0.3 other two fully connected layers come with RELUs.

The last layer contains only a single node, which provides the steering angle.

![Model Architecture][image1]

#### 2. Attempts to reduce overfitting in the model

This model is much bigger than neccessary. The evidence is that the training accuracy becomes lover than the validation accuracy as soon as the 2nd epoch. This is a clear sign of overtraining. Possible solutions:
* Feed it with more data
* Use dropout
* Lessen the network size
* Lessen the number of epochs

Because the model drove the car well, I decided to lessen the epochs to only 2. It passed my tests. So I did not feel to battle overtraining anymore. I added the dropout only because it was an expectation in the rubric points. (model.py line 50) 

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 57).

This is more or less my first iteration, and as it worked well, I didn't fine tune it too much. I tuned:
* the number of epochs to decrease overtraining,
* the dropout from the same reason
* and the steering offset of left and right image input in order to balance between nervous wobbling of the car and unabilty to steear strong.

Further fine tuning goals could be finding the minimum model size and lessening the left-right wobbling of the car.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road.

I used center driving on the 1st track, and race line driving on the 2nd track. 
I used 2 full round's data from driving on the 1st track and 2 full round's image data from driving on the 2nd track. I was driving with keyboard. When testing on the 2nd track in reverse it had some failures, so I added a reverse driving dataset as input later.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have the model analyze the images every frame and calculate a single output node that servers as steering input.

I previously had good experience with a convolutional network, so I thought that might be a good start. Upscaled it to fit the image size and applied bigger than 1 stride and maxpool 4 * 4 layers to decrease the image size as fast as possible while keeping the useful information at the same time.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I added more input images, dropout and changed the number of the epochs to only one.

The final step was to run the simulator to see how well the car was driving around track one. It run almost immediately well on the 1st track, but had problems with the 2nd one mainly probably due to error that seemed to be rooted between drive.py and the simulator. The car simply stopped on the center of plain road. Increasing the car speed to 14 solved the problem, but increased the challenge. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The initial model proved to be good, so I did not make any modifications to it. Applied only some fine tuning at the above written places.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:  

![Center image example][image2]

I collected all data in a ./data folder, and each dataset (run) in a subfolder 01, 02, and so on. This is big amount of data, so it is not uploaded to GitHub.

I also collected the left and right camera images, and modified their steering values by 0.3 and -0.3 respectively so that when the center camera sees something similar during playback, it knows that it should steer back.

To have more useful data I flipped the images and used both the original and flipped ones for training.

![Cropped and flipped images][image3]

I also cropped the top and bottom of the images and normalized the pixel values.

The above preprocessing proved to be sufficient to drive the 1st track both in forward and reverse direction, even though I did not collect reverse driving data. 

Then I decided to go for the 2nd track. I collected one lap of data in forward and one lap of data in reverse direction. This proved sufficient for the car to drive both tracks in both directions.

There was no need to capture recovey images.

My input data ended up consisting of ca. 27500 images.

I finally randomly shuffled the data-set and put 20% of the data into a validation set. 

The car was able to drive on both tracks and directions, but the driving was somewhat jerky. The main reason was that I drove by keyboard and my driving was also jerky. I decided to improve my driving data and repeated all runs with keyboard, and retrained the model. The result was not really surprising:
* the autonomous driving became much smoother,
* but at the same time it introduced a strong wobbling left to right.

The reason is probably the fact that the modell was not really faced with recovery situations, and did not learn the difference betweeen straight road and bendings. Or, alternatively, the incrased speed needs adjustements in the PI controller. Possible solutions could be collecting recovery driving data, and playing with those values. Due to time pressure I decided to fall back to the previously used data for now, which improved the drive quality.
