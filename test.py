import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

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

model = model_import.import_model(config)
modelPath = ".\\model\\" + config['model_name']
model.load_weights(modelPath)
		
######################################################################################
############### Add Tensorboard Support ############################

tensorboard = tb.TrainValTensorBoard()
tensorboard.set_model(model)

######################################################################################
################ Import Data ############################

if config['resample'] == "true":
	images, _ = data_import.import_data(config, isTrain=False)
else:
	images, _ = data_import.import_data(config, resample=False, isTrain=False)

######################################################################################
################ Patchwise Support ############################

if config['patch_wise'] == "true":
	if '2d' in config['network']:
		print()
################ 2D not finished yet ################
#		import patch_wise_2D as patch_wise
	elif '3d' in config['network']:
		import patch_wise_3D as patch_wise

######################################################################################
################ Test ############################




######################################################################################