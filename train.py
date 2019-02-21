import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
#from medpy.io import load, save

import warnings
warnings.filterwarnings("ignore")

if platform == "linux" or platform == "linux2":
	# linux
	sys.path.insert(0, os.getcwd()+'//lib')
elif platform == "darwin":
	# OS X
	sys.path.insert(0, os.getcwd()+'//lib')
elif platform == "win32":
	# Windows...
	sys.path.insert(0, os.getcwd()+'\\lib')

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
if platform == "linux" or platform == "linux2":
	# linux
	modelPath = ".//model//" + config['model_name']
elif platform == "darwin":
	# OS X
	modelPath = ".//model//" + config['model_name']
elif platform == "win32":
	# Windows...
	modelPath = ".\\model\\" + config['model_name']
		
######################################################################################
############### Add Tensorboard Support ############################

tensorboard = tb.TrainValTensorBoard()
tensorboard.set_model(model)

######################################################################################
################ Import Data ############################

if config['resample'] == "true":
	images, labels, _ = data_import.import_data(config, isTrain=True)
else:
	images, labels, _ = data_import.import_data(config, resample=False, isTrain=True)

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
################ Train ############################

learning_rate = float(config['learning_rate'])
metrics = [loss.dice_coef, 'acc']
decay_rate = float(config['decay_rate'])
batch_size = int(config['batch_size'])
dimension_size_x = int(config['dimension_size_x'])
dimension_size_y = int(config['dimension_size_y'])
dimension_size_z = int(config['dimension_size_z'])

for step, (image_batch, label_batch) in enumerate(patch_wise.generate_batch(images,labels,dimension_size_x,dimension_size_y,dimension_size_z,batch_size)):
    
	model.fit(x=image_batch, y=label_batch, batch_size=batch_size, epochs=2, verbose=1, validation_split=0.0, shuffle=True, initial_epoch=0)
	
	if step % 500 == 0:
	
		learning_rate *= decay_rate
		model.compile(optimizer=Adam(lr=learning_rate), loss=eval("loss."+config['loss']), metrics=metrics)
		
		model.save(model_path, overwrite=True)
		print("Model saved!")

######################################################################################