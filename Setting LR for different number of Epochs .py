# Setting LR for different number of Epochs 
def lr_schedule(epoch): 
    lr = 1e-3
    if epoch > 180: 
        lr *= 0.5e-3
    elif epoch > 160: 
        lr *= 1e-3
    elif epoch > 120: 
        lr *= 1e-2
    elif epoch > 80: 
        lr *= 1e-1
    print('Learning rate: ', lr) 
    return lr 