# ResNet V2 architecture 
def resnet_v2(input_shape, depth, num_classes = 10): 
    if (depth - 2) % 9 != 0: 
        raise ValueError('depth should be 9n + 2 (eg 56 or 110 in [b])') 
    # Start model definition. 
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9) 
  
    inputs = Input(shape = input_shape) 
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths 
    x = resnet_layer(inputs = inputs, 
                     num_filters = num_filters_in, 
                     conv_first = True) 
  
    # Instantiate the stack of residual units 
    for stage in range(3): 
        for res_block in range(num_res_blocks): 
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0: 
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage 
                    activation = None
                    batch_normalization = False
            else: 
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage 
                    strides = 2    # downsample 
  
            # bottleneck residual unit 
            y = resnet_layer(inputs = x, 
                             num_filters = num_filters_in, 
                             kernel_size = 1, 
                             strides = strides, 
                             activation = activation, 
                             batch_normalization = batch_normalization, 
                             conv_first = False) 
            y = resnet_layer(inputs = y, 
                             num_filters = num_filters_in, 
                             conv_first = False) 
            y = resnet_layer(inputs = y, 
                             num_filters = num_filters_out, 
                             kernel_size = 1, 
                             conv_first = False) 
            if res_block == 0: 
                # linear projection residual shortcut connection to match 
                # changed dims 
                x = resnet_layer(inputs = x, 
                                 num_filters = num_filters_out, 
                                 kernel_size = 1, 
                                 strides = strides, 
                                 activation = None, 
                                 batch_normalization = False) 
            x = keras.layers.add([x, y]) 
  
        num_filters_in = num_filters_out 
  
    # Add classifier on top. 
    # v2 has BN-ReLU before Pooling 
    x = BatchNormalization()(x) 
    x = Activation('relu')(x) 
    x = AveragePooling2D(pool_size = 8)(x) 
    y = Flatten()(x) 
    outputs = Dense(num_classes, 
                    activation ='softmax', 
                    kernel_initializer ='he_normal')(y) 
  
    # Instantiate model. 
    model = Model(inputs = inputs, outputs = outputs) 
    return model