import tensorflow as tf

def unet(inputs, out_channels, depth, num_fmaps, activation=None):
    """
    Implementation of
    U-Net: Convolutional Networks for Biomedical Image Segmentation    
    (Ronneberger et al., 2015)    
    https://arxiv.org/abs/1505.04597    
    Args:  
    inputs (keras input) 
    out_channels (int): number of output channels    
    depth (int): depth of the network    
    num_fmaps (int): number of filters in the first layer (doubled on each depth level)    
    """
    def conv_block(inputs, num_fmaps):
        nonlocal fov, sp
        x = tf.keras.layers.Conv2D(num_fmaps, 3, data_format='channels_first', padding='same')(inputs)    
        fov += ((3-1) * sp)
        sp *= 1
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(num_fmaps, 3, data_format='channels_first', padding='same')(x)    
        fov += ((3-1) * sp)
        sp *= 1
        x = tf.keras.layers.ReLU()(x)
        return x
    
    def add_encoder_down_block(inputs, num_fmaps): 
        nonlocal fov, sp
        x_left = conv_block(inputs, num_fmaps)
        x = tf.keras.layers.MaxPool2D(padding='same', data_format='channels_first')(x_left)
        fov += ((2-1) * sp)
        sp *= 2
        return x, x_left
        
    def add_decoder_up_block(inputs, x_left, num_fmaps):
        nonlocal sp
        x = tf.keras.layers.Conv2DTranspose(num_fmaps, kernel_size=2, strides=2, data_format='channels_first', padding='same')(inputs)
        sp //= 2
        x = tf.keras.layers.Concatenate(axis=1)([x, x_left])
        x = conv_block(x, num_fmaps)
        return x
        
    x = inputs
    fov = 1
    sp = 1
    skips = []
    for i in range(depth):
        x, x_left = add_encoder_down_block(x, num_fmaps)
        num_fmaps *= 2
        skips.append(x_left)
    x = conv_block(x, num_fmaps)
    fov += ((3-1) * sp)
    for i in range(depth-1, -1, -1):
        num_fmaps //= 2
        x = add_decoder_up_block(x, skips[i], num_fmaps)
            
    x = tf.keras.layers.Conv2D(out_channels, 1, data_format='channels_first', padding='same')(x)
    if activation is not None:
        x = activation(x)
    fov += (1-1) * sp
    sp *= 1
    return x, fov