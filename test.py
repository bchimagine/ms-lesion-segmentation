import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model
from medpy.io import load, save

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd()+'\lib')

############### Import Configuration ############################

import losses as loss
import tensorboard as tb
import data_import
import model_import

config = {}
with open("config.txt") as file:
	for line in file:
		(key, val) = line.split(':')
		val = val.replace("\"", "")
		val = val.strip()
		val = " ".join(val.split())
		key = key.strip()
		key = " ".join(key.split())
		config[key] = val
		
config['modified_format'] = "nii"
		
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu']
		
######################################################################################
############### Build Network Architecture ############################

	#### Build the model and load the weights after ####
	######## Have to know the exact structure of the trained model #########
	
#model = model_import.import_model(config)
#modelPath = ".\\model\\" + config['model_name']
#model.load_weights(modelPath)

	#### Or load the whole model and weights all together ####
	######## don't need to know anything about the trained model #########
modelPath = ".\\model\\" + config['model_name']
model = load_model(modelPath, custom_objects={'focal_loss': eval("loss.focal_loss"), 'dice': eval("loss.dice_loss"), 'tversky': eval("loss.fbeta_loss")})
		
######################################################################################
############### Add Tensorboard Support ############################

tensorboard = tb.TrainValTensorBoard()
tensorboard.set_model(model)

######################################################################################
################ Import Data ############################

if config['resample'] == "true":
	images, _, file_names = data_import.import_data(config, isTrain=False)
else:
	images, _, file_names = data_import.import_data(config, resample=False, isTrain=False)

######################################################################################
################ Patchwise Support ############################

if config['patch_wise'] == "true":
	if '2d' in config['network']:
################ 2D not tested yet ################
		import patch_wise_2D as patch_wise
	elif '3d' in config['network']:
		import patch_wise_3D as patch_wise

######################################################################################
################ Patch Fusion Support ############################

if config['patch_wise'] == "true":
	if '2d' in config['network']:
		import patch_fusion_2D as patch_fusion
	elif '3d' in config['network']:
		import patch_fusion_3D as patch_fusion

######################################################################################
################ Test ############################

for i in range(len(images)):
	
	predictions_smooth = patch_fusion.predict_img_with_smooth_windowing(
															images[i],
															window_size=int(config['dimension_size_x']), # Only supports square and cubic patch sizes for now
															subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
															nb_classes=int(config['classes']),
															pred_func=( lambda batch: model.predict(batch, batch_size=int(config['batch_size']), verbose=0) )
														)
														
	predictions_smooth[predictions_smooth >= 0.5] = 1
	predictions_smooth[predictions_smooth < 0.5] = 0
	
	save(predictions_smooth[0],file_names[i].replace('modified_data', 'prediction'))

######################################################################################