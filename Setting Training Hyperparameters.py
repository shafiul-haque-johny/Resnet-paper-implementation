# Setting Training Hyperparameters 
batch_size = 32  # original ResNet paper uses batch_size = 128 for training 
epochs = 200
data_augmentation = True
num_classes = 10
  
# Data Preprocessing  
subtract_pixel_mean = True
n = 3
  
# Select ResNet Version 
version = 1
  
# Computed depth of  
if version == 1: 
    depth = n * 6 + 2
elif version == 2: 
    depth = n * 9 + 2
  
# Model name, depth and version 
model_type = 'ResNet % dv % d' % (depth, version) 
  
# Load the CIFAR-10 data. 
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
  
# Input image dimensions. 
input_shape = x_train.shape[1:] 
  
# Normalize data. 
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
  
# If subtract pixel mean is enabled 
if subtract_pixel_mean: 
    x_train_mean = np.mean(x_train, axis = 0) 
    x_train -= x_train_mean 
    x_test -= x_train_mean 
  
# Print Training and Test Samples  
print('x_train shape:', x_train.shape) 
print(x_train.shape[0], 'train samples') 
print(x_test.shape[0], 'test samples') 
print('y_train shape:', y_train.shape) 
  
# Convert class vectors to binary class matrices. 
y_train = keras.utils.to_categorical(y_train, num_classes) 
y_test = keras.utils.to_categorical(y_test, num_classes) 