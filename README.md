# Flood Detection Through Imagery

By 

Scott Armstrong ([personal github](https://github.com/Eeyle), [linkedin](https://www.linkedin.com/in/sc-armstrong/), [GA enterprise github](https://git.generalassemb.ly/eeyle))

Faisal Kalema ([personal github](https://github.com/Kalz123), [linkedin](https://www.linkedin.com/in/faisalkalema/), [GA enterprise github](https://git.generalassemb.ly/FAISAL123))

Richard Ling ([personal github](https://github.com/rich808), [linkedin](https://www.linkedin.com/in/richardzling/), [GA enterprise github](https://git.generalassemb.ly/Rich88))

General Assembly DSI-11 2020 Project #5 (Client Project with New Light Technologies)

05/15/20

## Overview / Problem Statement
With the advanced technologies we currently have, pictures of anything can be spread very quickly around the world. In the event of a flood, we can see pictures of the food as it is happening. These pictures can show us the situation of the flood and its damages.  Thus, if these images can show a severity level about the flood, it can give a first-hand notice to those who may be impacted to have early preparation. This leads to our problem statement: Can we use images to detect flooding and its severity?

### Data Collection
#### New York Times API
Images were pulled via the [Article Search](https://developer.nytimes.com/docs/articlesearch-product/1/routes/articlesearch.json/get) entry point of the NYT API. All articles between 2004 and 2020 with the "Flood" subject tag were collected and their banner images were retrieved. In total about 600 images were found this way.
#### Gettyimages
By searching for “hurricane Katrina” from [getty images](https://www.gettyimages.com/editorial-images), we were able to obtain images related to flooding between pages 1 and 60 using BeautifulSoup. We scraped about 1000 images.

### Data Cleaning
In this section, we discarded all the images that were irrelevant or unrelated to the flood (for the image detection problem) and images where the object levels were not clear (e.g images from an aerial view).


![clean data example ](https://git.generalassemb.ly/eeyle/client-project/blob/master/label_examples/data_clean.png)

For the classification problem however, we kept both images related to the flood and other unrelated random images so that we could test on non-flood images as well.

#### Labeling
Using the [labelImg](https://github.com/tzutalin/labelImg) application, Four levels (level 1, level 2, level 3, level 4) were assigned to 3 different types of objects: Human beings, Cars and Houses; where level 1 represents a deep flood, which is about 60 inches and above for an average human being, Level 2 is the waist level that represents an average flood, level 3 is the knee level and level 1 being the shallowest flood around the ankle. The same idea for the cars and houses was applied. 

Label Criteria for car and human:

![label criteria for car and human ](https://git.generalassemb.ly/eeyle/client-project/blob/master/label_examples/label_criteria.png)


Label Examples:

School Bus at level 1:

![school bus level 1 ](https://git.generalassemb.ly/eeyle/client-project/blob/master/label_examples/school_bus.png)

People at level 2:
![people level 2 ](https://git.generalassemb.ly/eeyle/client-project/blob/master/label_examples/people.png)


### Flood Detection Model
At the time of this project, we couldn’t fully build an image detection model due to time constraints plus we had not yet covered neural networks. Therefore, we used a pre-trained model by [Joseph Nelson](https://www.linkedin.com/in/josephofiowa/) called [RoboFlow](https://models.roboflow.ai/object-detection/yolo-v3-pytorch) in [Google Colab](https://colab.research.google.com/drive/1ntAL_zI68xfvZ4uCSAF6XT27g0U4mZbW#scrollTo=IaVwHzdprdSN) to execute the concepts and ideas of this problem statement.  
This Flood Detection Model will detect the object we label with its label. To improve the training model and expose the images to different exposure, we augmented the images by brightening and darkening by 40%.
302 images were resized to 412x412 and fed into the model Each image was mapped 3 times with different exposures and ended up with a total of 906 images after augmentation.

### Flood Classification Model
The second model was a binary classifier that determined whether or not a given image is flooded. The model takes in a 32x32 grayscale image as its input. Since the model intended to distinguish between flood and non-flood images, it was fed both flood and non-flood images for training, so many images that were not directly showing a flood were kept.

A similar image augmentation process was used for training classification. Images were resized to 96x96 and converted to grayscale. Images were then mirrored over the vertical axis, and five different brightness levels were used for every image. Finally, partially-overlapping sub-images of size 32x32 were taken from the image to augment it further. This process began with 908 original images and increased the data by 160-fold, ending with 145,280 images.

Several models were tried, but the only model that achieved the best performance was a MLP neural network. The model performed the best with the widest possible base, so the first hidden layer contained 128 nodes compared to the 1024 input features. The second hidden layer contained 16 nodes before collapsing into the result of flooded or not flooded. 

The model can be found encapsulated in the class [FloodImageClassifier.py](https://git.generalassemb.ly/eeyle/client-project/blob/master/code/FloodImageClassifier.py) in the `code` directory. This class implements a `predict` method but no `fit` method nor others, so it's not a proper Scikit-Learn estimator. The class loads a pre-fit version of the model, saved to a separate file [binary_images/flood_image_classifier_mlp_128_16.sav](https://git.generalassemb.ly/eeyle/client-project/blob/master/binary_images/flood_image_classifier_mlp_128_16.sav).

### Results
#### Flood Detection Model
In a total of 60 pictures, it was able to detect objects in 48 of the images due to insufficient images to the train data. In those 48 images, it labeled a total of 88 objects and correctly labeled 66 of those objects.  

![pic 51](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic51.png)

The model was able to detect the bus was partially covered, although it sunk into the ground and not covered by water.

![pic 54](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic54.png)

The model correctly detected each individual with water at around their ankles.

![pic 30](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic30.png)

The models correctly detected the car at the side of the image but misclassified the person in the middle. Although it was misclassified, it was within reasonable range of level 4 and not level 2.  

![pic 22](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic22.png)

It correctly detected the unflooded dome but not the structure below the dome because we didn’t have enough images for the complete dome. 
For the people, it was difficult for human eyes to visualize the levels but it was still able to assign a reasonable level to them, considering the water was somewhere between the waist and the knee of the individual. 

![pic 49](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic49.png)

Due to the lack of images to train the model to recognize the dome, it wasn’t able to detect the dome in this image.
The cars were correctly classified with correct levels. 

![pic 29](https://git.generalassemb.ly/eeyle/client-project/blob/master/results_pic/pic29.png)

The person on the left was misclassified, possibly due to the reflection on the water. 
The person in the middle was misclassified but correct in its context because we didn’t train the model to recognize the boat. 
The person on the right of the image was correctly classified at level 3 with water between the ankle and knees.   

#### Flood Classification Model
When validating on purely 32x32 sub-images, the model was able to achieve a notable level of success, with an accuracy of ~80% compared to a base accuracy of 60%. With further training and model iteration, we are confident that a classification model can reliably distinguish between flooded and non-flooded images.

When testing, however, the model was not as accurate. This was expected because brand new test data should essentially always reduce the model's accuracy. Furthermore every test image was tested as a whole image, rather than as 160 sub-images, meaning the test sample was considerably smaller, and meaning that the model's bias should have been averaged out quite a bit since the result of a test image is the average of the results of its 160 sub-images. The test model achieved an accuracy of ~65% compared to a base test accuracy of 55%.

It's suspected that the model is picking up on large amounts of water in the bottom half of the image.

 ![Test image 25](https://git.generalassemb.ly/eeyle/client-project/blob/master/binary_images/test_images/img_test_25.jpg) 

The above image is a good example of this, where the model classifies the ocean as being a flood. Images where the floodwater is located in the top half of the image such as the following image

![test image 35](https://git.generalassemb.ly/eeyle/client-project/blob/master/binary_images/test_images/img_test_35.jpg) 

were particularly prone to being false negatives. Overall though, the model did well at distinguishing when an image was of a person/house/dog and when an image was of a proper flood. The model will perform extremely poorly when distinguishing between boats/oceans/rivers and floods, but is decent at distinguishing between people/animals and floods, meaning that it could be used to filter data that has been mass-scraped from social media as long as the scraping was already searching for flooding.

### Conclusion/Future Improvements
The object detection model shows us that we can use images to approximate the depth of the flood by giving proper labeling to each object. To further improve this concept or this model, it needs more images of different objects to train the model to detect objects with higher confidence levels. 

We have shown that a MLP classification model, with enough processor and programmer time, could reliably distinguish between whether an image is flooded or not.

In practice, these two models can be used in combination with each other to more accurately determine flood depths: once a large number of social media images have been scraped, unwanted images can be filtered out by the classification model and so that only the most relevant photos can be used for the flood detection model.
