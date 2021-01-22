# Basic ResNet Building Block 
def resnet_layer(inputs, 
                 num_filters = 16, 
                 kernel_size = 3, 
                 strides = 1, 
                 activation ='relu', 
                 batch_normalization = True, 
    conv = Conv2D(num_filters, 
                  kernel_size = kernel_size, 
                  strides = strides, 
                  padding ='same', 
                  kernel_initializer ='he_normal', 
                  kernel_regularizer = l2(1e-4)) 
  
    x = inputs 
    if conv_first: 
        x = conv(x) 
        if batch_normalization: 
            x = BatchNormalization()(x) 
        if activation is not None: 
            x = Activation(activation)(x) 
    else: 
        if batch_normalization: 
            x = BatchNormalization()(x) 
        if activation is not None: 
            x = Activation(activation)(x) 
        x = conv(x) 
    return x