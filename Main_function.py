# Main function  
if version == 2: 
    model = resnet_v2(input_shape = input_shape, depth = depth) 
else: 
    model = resnet_v1(input_shape = input_shape, depth = depth) 
  
model.compile(loss ='categorical_crossentropy', 
              optimizer = Adam(learning_rate = lr_schedule(0)), 
              metrics =['accuracy']) 
model.summary() 
print(model_type) 
  
# Prepare model model saving directory. 
save_dir = os.path.join(os.getcwd(), 'saved_models') 
model_name = 'cifar10_% s_model.{epoch:03d}.h5' % model_type 
if not os.path.isdir(save_dir): 
    os.makedirs(save_dir) 
filepath = os.path.join(save_dir, model_name) 
  
# Prepare callbacks for model saving and for learning rate adjustment. 
checkpoint = ModelCheckpoint(filepath = filepath, 
                             monitor ='val_acc', 
                             verbose = 1, 
                             save_best_only = True) 
  
lr_scheduler = LearningRateScheduler(lr_schedule) 
  
lr_reducer = ReduceLROnPlateau(factor = np.sqrt(0.1), 
                               cooldown = 0, 
                               patience = 5, 
                               min_lr = 0.5e-6) 
  
callbacks = [checkpoint, lr_reducer, lr_scheduler] 
  
# Run training, with or without data augmentation. 
if not data_augmentation: 
    print('Not using data augmentation.') 
    model.fit(x_train, y_train, 
              batch_size = batch_size, 
              epochs = epochs, 
              validation_data =(x_test, y_test), 
              shuffle = True, 
              callbacks = callbacks) 
else: 
    print('Using real-time data augmentation.') 
    # This will do preprocessing and realtime data augmentation: 
    datagen = ImageDataGenerator( 
        # set input mean to 0 over the dataset 
        featurewise_center = False, 
        # set each sample mean to 0 
        samplewise_center = False, 
        # divide inputs by std of dataset 
        featurewise_std_normalization = False, 
        # divide each input by its std 
        samplewise_std_normalization = False, 
        # apply ZCA whitening 
        zca_whitening = False, 
        # epsilon for ZCA whitening 
        zca_epsilon = 1e-06, 
        # randomly rotate images in the range (deg 0 to 180) 
        rotation_range = 0, 
        # randomly shift images horizontally 
        width_shift_range = 0.1, 
        # randomly shift images vertically 
        height_shift_range = 0.1, 
        # set range for random shear 
        shear_range = 0., 
        # set range for random zoom 
        zoom_range = 0., 
        # set range for random channel shifts 
        channel_shift_range = 0., 
        # set mode for filling points outside the input boundaries 
        fill_mode ='nearest', 
        # value used for fill_mode = "constant" 
        cval = 0., 
        # randomly flip images 
        horizontal_flip = True, 
        # randomly flip images 
        vertical_flip = False, 
        # set rescaling factor (applied before any other transformation) 
        rescale = None, 
        # set function that will be applied on each input 
        preprocessing_function = None, 
        # image data format, either "channels_first" or "channels_last" 
        data_format = None, 
        # fraction of images reserved for validation (strictly between 0 and 1) 
        validation_split = 0.0) 
  
    # Compute quantities required for featurewise normalization 
    # (std, mean, and principal components if ZCA whitening is applied). 
    datagen.fit(x_train) 
  
    # Fit the model on the batches generated by datagen.flow(). 
    model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 
                        validation_data =(x_test, y_test), 
                        epochs = epochs, verbose = 1, workers = 4, 
                        callbacks = callbacks) 
  
# Score trained model. 
scores = model.evaluate(x_test, y_test, verbose = 1) 
print('Test loss:', scores[0]) 
print('Test accuracy:', scores[1])