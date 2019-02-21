# Lesion Segmentation

### Config file options

    network: "2d-unet", "3d-unet", "2d-dense", "3d-dense", "3d-dense-compact"
	
    loss: "focal_loss", "fbeta_loss", "dice_loss", "categorical_crossentropy", "generalized_dice_loss"
	
	learning_rate: ..., "0.01", "0.005", "0.001", ...
	
	decay_rate: "0.95", "0.9", ...
	(decay for learning rate after each 500 epochs)
	
	gpu: "0", "1", "2", "3", "0,1", "0,2", ..., "0,1,2", "0,1,3", ..., "0,1,2,3"
	(selection of GPUs to be visible to be program)
	
	dimension_size_X: ..., "64", "128", "256", "512", ...
	(dimension sizes of the patches)
	
	final_image_shape: (XXX,XXX,XX)
	(shape of the disired resampled images - no effect if "resample" is "false")
	
	sequences: "1", ...
	(number of image modalities)
	
	classes: "1", ...
	(number of prediction classes)
	
	batch_size: "1", ...
	(number of batches for training)
	
	raw_format: "nrrd" or "nii"
	(original format of images)
	
	resample: "true" or "false"
	
	patch_wise: "true" or "false"
	
	model_name: "XXXXX.hdf5" or "XXXXX.hd5"
	(model name located in the model folder)
	
### Tree structure of files and directories

- root
	- data
		- modified_data		(Put your data here, only if resample is False and you don't want to resample)
		- predictiona		(Your predictions will be saved here)
		- raw_data			(Put your data here, only if resample is True and you want to resample)
			- XXX_000.nii
			- XXX_001.nii
			- XXX_002.nii
			- ...
	- lib
		- All (.py) library files (You might have to install some other python packages too, like tensorflow, keras, medpy, etc)
	- model
		- trained models with formats of (XXX.hd5 or XXX.hdf5)
	- logs
- config
- data_import
- model_import
- test.py
- train.py

