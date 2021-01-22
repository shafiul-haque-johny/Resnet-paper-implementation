#ResNet V1 architecture  
def resnet_v1(input_shape, depth, num_classes = 10): 
      
    if (depth - 2) % 6 != 0: 
        raise ValueError('depth should be 6n + 2 (eg 20, 32, 44 in [a])') 
    # Start model definition. 
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6) 
  
    inputs = Input(shape = input_shape) 
    x = resnet_layer(inputs = inputs) 
    # Instantiate the stack of residual units 
    for stack in range(3): 
        for res_block in range(num_res_blocks): 
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack 
                strides = 2  # downsample 
            y = resnet_layer(inputs = x, 
                             num_filters = num_filters, 
                             strides = strides) 
            y = resnet_layer(inputs = y, 
                             num_filters = num_filters, 
                             activation = None) 
            if stack > 0 and res_block == 0:  # first layer but not first stack 
                # linear projection residual shortcut connection to match 
                # changed dims 
                x = resnet_layer(inputs = x, 
                                 num_filters = num_filters, 
                                 kernel_size = 1, 
                                 strides = strides, 
                                 activation = None, 
                                 batch_normalization = False) 
            x = keras.layers.add([x, y]) 
            x = Activation('relu')(x) 
        num_filters *= 2
  
    # Add classifier on top. 
    # v1 does not use BN after last shortcut connection-ReLU 
    x = AveragePooling2D(pool_size = 8)(x) 
    y = Flatten()(x) 
    outputs = Dense(num_classes, 
                    activation ='softmax', 
                    kernel_initializer ='he_normal')(y) 
  
    # Instantiate model. 
    model = Model(inputs = inputs, outputs = outputs) 
    return model 