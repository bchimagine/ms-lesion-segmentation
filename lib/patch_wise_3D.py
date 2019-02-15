
import numpy as np

####################### PATCH WISE #################################

def generate_batch(train_images,train_label, dimension_size_x, dimension_size_y, dimension_size_z, batch_size = 1):
    
    images = train_images
    labels = train_label
    
    width = int(dimension_size_x)
    height = int(dimension_size_y)
    depth = int(dimension_size_z)
    
    counter = 0    
    
    for samples in generate_samples(np.shape(train_images)[0], batch_size):
			
        image_batch = images[samples, :, :, :, :]
        label_batch = labels[samples, :, :, :, :]                        
        
        if counter < 100:
            if np.count_nonzero(label_batch[0,:,:,:,0]) < 100 or \
                np.count_nonzero(image_batch[0,:,:,:,0]) == 0:
                continue
                
        else:
            if np.count_nonzero(label_batch[0,:,:,:,0]) < 10 or \
                np.count_nonzero(image_batch[0,:,:,:,0]) == 0:
                continue
        
        LesiorArg = list(zip(*np.where(labels[samples,:,:,:,0] == 1)))
        np.random.shuffle(LesiorArg)
        
        while True:

            rand = np.random.randint(len(LesiorArg))

            x_index = LesiorArg[rand][0]
            y_index = LesiorArg[rand][1]
            z_index = LesiorArg[rand][2]
            
            x_index += np.random.randint(-20, 20)
            y_index += np.random.randint(-20, 20)
            z_index += np.random.randint(-20, 20)
            
            if x_index > np.shape(images)[1] - width//2:
                x_index = np.shape(images)[1] - width//2
            elif x_index < width//2:
                x_index = width//2
                
            if y_index > np.shape(images)[2] - height//2:
                y_index = np.shape(images)[2] - height//2
            elif y_index < height//2:
                y_index = height//2
                
            if z_index > np.shape(images)[3] - depth//2:
                z_index = np.shape(images)[3] - depth//2
            elif z_index < depth//2:
                z_index = depth//2

            image_batch = images[samples, int(x_index-width/2):int(x_index+width/2), int(y_index-height/2):int(y_index+height/2), int(z_index-height/2):int(z_index+height/2)]
            label_batch = labels[samples, int(x_index-width/2):int(x_index+width/2), int(y_index-height/2):int(y_index+height/2), int(z_index-height/2):int(z_index+height/2)]
            
            if counter < 1000 and np.count_nonzero(label_batch[0,int(width/4):int(3*width/4),int(height/4):int(3*height/4),int(depth/4):int(3*depth/4),0]) < 50:
                continue
            elif counter >= 1000 and np.count_nonzero(label_batch[0,int(width/4):int(3*width/4),int(height/4):int(3*height/4),int(depth/4):int(3*depth/4),0]) < 5:
                continue
            else:
                break
        
        ###########################
            
        for i in range(image_batch.shape[0]):
            image_batch[i], label_batch[i] = augment_sample(image_batch[i], label_batch[i])
            
        counter += 1
            
        yield(image_batch, label_batch)

def generate_samples(NumberOfSamples, batch_size):
    n_samples = NumberOfSamples
    n_epochs = 10000
    n_batches = n_samples/batch_size
    for _ in range(n_epochs):
        sample_ids = np.random.permutation(n_samples)
        for i in range(int(n_batches)):
            inds = slice(i*batch_size, (i+1)*batch_size)
            yield sample_ids[inds]

def augment_sample(image, label):
    # Flipping
    if np.random.randint(4) == 1:
        image = np.rot90(image, 2, (0,1))
        label = np.rot90(label, 2, (0,1))
    
    elif np.random.randint(4) == 2:
        image = np.rot90(image, 2, (0,2))
        label = np.rot90(label, 2, (0,2))
    
    elif np.random.randint(4) == 3:
        image = np.rot90(image, 2, (1,2))
        label = np.rot90(label, 2, (1,2))
    
    return(image, label)