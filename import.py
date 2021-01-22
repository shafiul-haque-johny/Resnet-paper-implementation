# Import Keras modules and its important APIs 
import keras 
from keras.layers import Dense, Conv2D, BatchNormalization, Activation 
from keras.layers import AveragePooling2D, Input, Flatten 
from keras.optimizers import Adam 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
from keras.callbacks import ReduceLROnPlateau 
from keras.preprocessing.image import ImageDataGenerator 
from keras.regularizers import l2 
from keras import backend as K 
from keras.models import Model 
from keras.datasets import cifar10 
import numpy as np 
import os 