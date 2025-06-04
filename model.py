import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, UpSampling2D, Concatenate, AveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB7

def AC_Block(x, filter):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filter, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_efficientunet(input_shape=(256, 256, 3)):
    # Input
    inputs = Input(input_shape)
    
    # Encoder (EfficientNetB7)
    base_model = EfficientNetB7(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Encoder features
    s1 = base_model.get_layer('input_1').output
    s2 = base_model.get_layer('block2a_expand_activation').output
    s3 = base_model.get_layer('block3a_expand_activation').output
    s4 = base_model.get_layer('block5a_expand_activation').output
    
    # Bridge
    b1 = base_model.get_layer('top_activation').output
    
    # Apply AC Block to the bridge
    b1 = AC_Block(b1, 320)  # 320 is the number of filters in the last layer of EfficientNetB7
    
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    
    # Output
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)
    
    model = Model(inputs, outputs, name="EfficientU-Net")
    return model

if __name__ == "__main__":
    model = build_efficientunet()
    model.summary()
