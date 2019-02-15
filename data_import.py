
################## Resmpling Packages and functions ##########################

from keras.layers import Input, Convolution2D, BatchNormalization, \
    Activation, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import densenet3DD as densenet
import numpy as np
from medpy.io import load, save

import SimpleITK as sitk

import os

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    
    original_spacing = itk_image.GetSpacing()
    
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)
	
##########################################################################################

############## Resampling ########################

def resample(dataPath, origFormat, newFormat):

	inputCounter = 0

	target_shape = (512, 512, 128)

	for root, dirs, files in os.walk(dataPath):
		
		print()		
		print("Reading from", root)
		print()
		modified = root.replace("raw_", "modified_")
		print("Writing to", modified)
		print()

		for fileName in files:

			if (fileName.endswith(origFormat)):
			
				is_label = True if 'label' in fileName else False

				print(root+'/'+fileName)

				print('--------------------------------------------')
				temp = sitk.ReadImage(root+'/'+fileName, sitk.sitkFloat32)
				
				print(temp.GetSize())
				print(temp.GetSpacing())

				image_shape = temp.GetSize()
				image_spacing = temp.GetSpacing()

				taget_spacing = [(1. * image_shape[0]*(image_spacing[0]/target_shape[0])),
								(1. * image_shape[1]*(image_spacing[1]/target_shape[1])),
								(1. * image_shape[2]*(image_spacing[2]/target_shape[2]))]

				temp = resample_image(temp, out_spacing=taget_spacing, is_label=is_label)

				print(temp.GetSize())
				print(temp.GetSpacing())
				print('--------------------------------------------')

				print(modified+'/'+fileName.replace(origFormat,newFormat))
				sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(temp), sitk.sitkFloat32),
								modified+'/'+fileName.replace(origFormat,newFormat))
							
#########################################################################################

############## Data Import ########################

def import_data(config, resample=True, isTrain=True):

	origFormat, newFormat = config['raw_format'], config['modified_format']
	
	dataPath = ".\\data\\raw_data\\"
	
	if resample == True:
		resample(dataPath, origFormat, newFormat)
	
	dataPath = ".\\data\\modified_data\\"
	
	print()
	
	imgs_final = []
	labels = []
	file_names = []
	
	for root, dirs, files in os.walk(dataPath):

		for fileName in sorted(files):
		
			if fileName.endswith('000.'+newFormat) and "label" not in fileName:

				print(root+'/'+fileName)
				file_names.append(root+'/'+fileName)
				
				###### Load Image ###############
				image_data_all = []
				for seq in range(int(config['sequences'])):
					image_data, image_header = load(root + '/' + fileName.replace('0.', str(seq)+'.'))
					
					image_data = np.expand_dims(image_data, axis=0)
					
					if '2d' in config['network']:
						image_data = np.swapaxes(image_data,0,3)
					elif '3d' in config['network']:
						image_data = np.expand_dims(image_data, axis=4)
						
					if len(image_data_all) == 0:
						image_data_all = image_data[:]
					else:
						if '2d' in config['network']:
							image_data_all = np.concatenate((image_data_all,image_data), 3)
						elif '3d' in config['network']:
							image_data_all = np.concatenate((image_data_all,image_data), 4)
				
				print(np.shape(image_data_all))
				
				if len(imgs_final) == 0:
					imgs_final = image_data_all[:]
				else:
					imgs_final = np.concatenate((imgs_final,image_data_all), 0)
	
				if isTrain:
					####### Load Label ##############
					label_data, label_header = load((root + '/' + fileName).replace('000.' + newFormat, 'label.' + newFormat))
					
					label_data = np.expand_dims(label_data, axis=0)
				
					if '2d' in config['network']:
						label_data = np.swapaxes(label_data,0,3)
					elif '3d' in config['network']:
						label_data = np.expand_dims(label_data, axis=4)
					
					
					if len(label_data[label_data == 1]) == 0:
						label_data[label_data > 0] = 1.
					
					print(np.shape(label_data))
						
					if len(labels) == 0:
						labels = label_data[:]
					else:
						labels = np.concatenate((labels,label_data), 0)
						
	imgs_final = np.asarray(imgs_final)
	labels = np.asarray(labels)
			
	print(np.shape(imgs_final))
	print(np.shape(labels))
				
	return (imgs_final, labels, file_names)
            