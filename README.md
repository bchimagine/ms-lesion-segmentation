# Lesion Segmentation

### Config file options:

    network: "2d-unet", "3d-unet", "2d-dense", "3d-dense", "3d-dense-compact"
	
    loss: "focal_loss", "fbeta_loss", "dice_loss", "categorical_crossentropy", "generalized_dice_loss"
	
	learning_rate: ..., "0.01", "0.005", "0.001", ...
	
	gpu: "0", "1", "2", "3", "0,1", "0,2", ..., "0,1,2", "0,1,3", ..., "0,1,2,3"
	
	dimension_size_X: ..., "64", "128", "256", "512", ...
	
	sequences: "1", ...
	
	classes: "1", ...
	
	raw_format: "nrrd" or "nii"
	
	modified_format: "nrrd" or "nii"