		
import losses as loss
from keras.optimizers import Adam
		
def import_model(config):
		
	######################################################################################
	############### Build Network Architecture ############################

	losses = ['focal_loss', 'fbeta_loss', 'dice_loss', 'generalized_dice_loss', 'categorical_crossentropy']

	if config['network'] == '2d-unet':
		import unet as unet
	elif config['network'] == '3d-unet':
		import unet3D as unet
	elif config['network'] == '2d-dense':
		import densenet as densenet
	elif config['network'] == '3d-dense':
		import densenet3D as densenet
	elif config['network'] == '3d-dense-compact':
		import densenet3DD as densenet
	else:
		print("Selected Network Architecture", config['network'], "not recognized.")
		sys.exit()
		
	dimension_size_x = int(config['dimension_size_x'])
	dimension_size_y = int(config['dimension_size_y'])
	dimension_size_z = int(config['dimension_size_z'])
	channels = int(config['sequences'])
	classes = int(config['classes'])

	if config['loss'] in losses:
		if 'unet' in vars() or 'unet' in globals():
		############### Build the Unet Model #######################

			network = unet()
			
			if '2d' in config['network']:
				model = network.model(dimension_size_x, dimension_size_y, loss=config['loss'], learning_rate=float(config['learning_rate']))
			elif '3d' in config['network']:
				model = network.model(dimension_size_x, dimension_size_y, dimension_size_z, loss=config['loss'], learning_rate=float(config['learning_rate']))
					
		elif 'densenet' in vars() or 'densenet' in globals():
		############### Build the DenseNet Model #######################
		
			if '2d' in config['network']:
				model = densenet.DenseNetFCN((dimension_size_x, dimension_size_y, channels), classes=1, growth_rate=16, nb_dense_block=5, 
										nb_layers_per_block=5, reduction=0.5, dropout_rate=0.2, upsampling_type='deconv', activation='sigmoid')
			elif '3d' in config['network']:
				model = densenet.DenseNetFCN((dimension_size_x, dimension_size_y, dimension_size_z, channels), classes=1, growth_rate=16, nb_dense_block=5, 
										nb_layers_per_block=5, reduction=0.5, dropout_rate=0.2, upsampling_type='deconv', activation='sigmoid')
			
			metrics = [loss.dice_coef, 'acc']
			model.compile(optimizer=Adam(lr=float(config['learning_rate'])), loss=eval("loss."+config['loss']), metrics=metrics)
	else:
		print("Selected Loss Function", config['loss'], "not recognized.")
		sys.exit()
			
	print("Number of Network Parameters:", model.count_params())
			
	######################################################################################
	
	return model