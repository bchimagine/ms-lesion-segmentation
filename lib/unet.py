from keras.layers import Input, Convolution2D, BatchNormalization, \
    Activation, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf

class Unet:
    def __init__(self):
        print('initialized calss')

    def conv_bn_relu(self, x, nb_filter, kernel_dim1, kernel_dim2):
        conv = Convolution2D(nb_filter, (kernel_dim1, kernel_dim2),
                             kernel_initializer='he_normal',
                             activation=None,
                             padding='same',
                             kernel_regularizer=l2(1e-5),
                             bias_regularizer=None,
                             activity_regularizer=None)(x)
        norm = BatchNormalization(axis=-1)(Dropout(0.2)(conv))
        x = Activation("elu")(norm)
        return x

    def root_1(self, x):
        conv1 = self.conv_bn_relu(x, 8, 3, 3)
        conv1 = self.conv_bn_relu(conv1, 8, 3, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = self.conv_bn_relu(pool1, 16, 3, 3)
        conv2 = self.conv_bn_relu(conv2, 16, 3, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_bn_relu(pool2, 32, 3, 3)
        conv3 = self.conv_bn_relu(conv3, 32, 3, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_bn_relu(pool3, 64, 3, 3)
        conv4 = self.conv_bn_relu(conv4, 64, 3, 3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv4_1 = self.conv_bn_relu(pool4, 128, 3, 3)
        conv4_1 = self.conv_bn_relu(conv4_1, 128, 3, 3)
        pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

        conv5 = self.conv_bn_relu(pool4_1, 256, 3, 3)
        conv5 = self.conv_bn_relu(conv5, 256, 3, 3)

        up5_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1], axis=-1)
        conv5_1 = self.conv_bn_relu(up5_1, 128, 3, 3)
        conv5_1 = self.conv_bn_relu(conv5_1, 128, 3, 3)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5_1), conv4], axis=-1)
        conv6 = self.conv_bn_relu(up6, 64, 3, 3)
        conv6 = self.conv_bn_relu(conv6, 64, 3, 3)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = self.conv_bn_relu(up7, 32, 3, 3)
        conv7 = self.conv_bn_relu(conv7, 32, 3, 3)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = self.conv_bn_relu(up8, 16, 3, 3)
        conv8 = self.conv_bn_relu(conv8, 16, 3, 3)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = self.conv_bn_relu(up9, 8, 3, 3)
        conv9 = self.conv_bn_relu(conv9, 8, 3, 3)

        conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
        return conv10

    def model(self):
        image = Input((dimension_size, dimension_size, 1))
        # output shape (180, 180, 1)
        softmax_output = self.root_1(image)
        model = Model(inputs=image, outputs=softmax_output)
        metrics = [dice_coef, 'acc']
        model.compile(optimizer=Adam(lr=0.01), loss=focal_loss,
                      metrics=metrics)
        return model