from .Model import Model
from ..util import STATE_TYPE_OPTION

from keras.layers import *


class SimpleConvNet(Model):
    def __init__(self, args, input_layer):
        """
        Defines a simple series of convolutional layers
        """

        self.args = args
        self.input = input_layer

        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(input_layer)))         # batch_size  x board_x x board_y x num_channels
        #h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(h_conv1)))         # batch_size  x board_x x board_y x num_channels
        #h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='same')(h_conv2)))        # batch_size  x (board_x-2) x (board_y-2) x num_channels
        #h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 3, padding='valid')(h_conv3)))        # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv1)
        s_fc1 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))                # batch_size x 1024
        s_fc2 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))                        # batch_size x 1024

        self.output = s_fc2
