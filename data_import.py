
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

	target_shape = (512, 512, 90)

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

def import_data(config):

	origFormat, newFormat = config['raw_format'], config['modified_format']
	
	dataPath = ".\\data\\raw_data\\"
	
	resample(dataPath, origFormat, newFormat)