
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageOps import mirror
from PIL.ImageEnhance import Brightness
import pickle

class FloodImageClassifier:
	'''
	Classifies an image into either flooded or non-flooded. This classifier uses a trained MLP neural network to classify an image based on the majority vote of 160 sub-images.
	'''
	
	def __init__(self, path_to_model, path_to_images_, threshold=0.5):
		'''
		Creates a new FloodImageClassifier object.
		Inputs:
			path_to_model: The file path to the saved MLP model. The trained model is saved to a different file and can be found in binary_images/flood_image_classifier_mlp_128_16.sav. 
			path_to_images_: The path to the directory containing the images that are intended to be classified.
			threshold: Threshold for what is and is not considered a flood.
		'''
		self.model = pickle.load(open(path_to_model, 'rb'))
		self.path_to_images = path_to_images_
		self.threshold = threshold
	
	
	def predict(self, X):
		'''
		Predicts flood or no flood on a series of image paths.
		Inputs:
			X: a Pandas Series where each element is the path to an image file within the path_to_images parameter.
		Outputs:
			A Series of predictions for the corresponding images, where 1 is flooded and 0 is not flooded.
		'''
		return X.apply(self.predict_x)
	
	
	def predict_x(self, image_file_name):
		'''
		Predicts flood or no flood on a single image.
		Inputs:
			image_file_name: The name of the image file to open within the path_to_images directory.
		Outputs:
			1 if the majority of the image is flooded, 0 otherwise.
		'''
		img = Image.open(
			self.path_to_images + image_file_name
		).convert('RGB').convert('L')
		
		imgs = self.augment_image(img)
		
		results = self.predict_images(imgs)
		
		return ((sum(results) / len(results)) > self.threshold)
	
	
	def augment_image(self, image):
		'''
		Turns one image into a list of 160, 32x32 greyscale images.
		Inputs:
			image: The image to be augmented as a PIL Image object.
		Outputs:
			A list of 160 PIL Image objects, each 32x32 and greyscale.
		'''
		# downsample the image
		img_resized = image.resize((96, 96), Image.BILINEAR)
		
		# reverse the image
		imgs_reversed = [img_resized.copy(), mirror(img_resized.copy())]
		
		# brighten and darken the image
		imgs_enhanced = []
		enhance_amounts = [0.5, 0.75, 1.0, 1.25, 1.5]
		for img_reversed in imgs_reversed:
		    img_enhancer = Brightness(img_reversed)
		    for enhance_amount in enhance_amounts:
		        imgs_enhanced.append(
		            img_enhancer.enhance(enhance_amount)
		        )
		
		# scan along the image to create several sub-images at 32x32
		imgs_scanned = []
		for img_enhanced in imgs_enhanced:
		    for i in range(4):
		        for j in range(4):
		            imgs_scanned.append(
		                img_enhanced.crop((16*i, 16*j, 32+16*i, 32+16*j))
		            )
		
		return imgs_scanned
		
	def predict_images(self, images):
		'''
		Takes a list of sub-images and predicts whether each sub-image is flooded or not.
		Inputs:
			images: A list of 32x32 greyscale PIL Image objects, returned by augment_image().
		Outputs:
			A list of binary values representing whether the model predicts each sub-image to be flooded or not.
		'''
		y = []
		for image in images:
			y.append(
				self.model.predict(
					pd.DataFrame(list(image.getdata())).T
				)[0]
			)
		return y
		
		
		
		
		
		
