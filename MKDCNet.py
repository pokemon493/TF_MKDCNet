import tensorflow as tf # tensorflow version 2.8.0
import keras
from keras import layers
# import tensorflow_addons as tfa
from keras.applications import resnet

'''
Unofficial tensorflow code implementation of paper "Automatic Polyp Segmentation with Multiple Kernel Dilated Convolution Network"
Paper link: https://arxiv.org/pdf/2206.06264v2.pdf
Offical pytorch code implementation: https://github.com/nikhilroxtomar/MKDCNet
I implemented the tf version code according to the official pytorch code as much as possible
'''

class Conv2D(layers.Layer):
    def __init__(self, out_c, kernel_size=3, padding='same', dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = keras.models.Sequential([
            layers.Conv2D(
                out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation_rate=dilation,
                use_bias=bias
            ),
            layers.BatchNormalization(),
        ])
        self.relu = layers.Activation('relu')
    
    def call(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class ResidualBlock(layers.Layer):
    def __init__(self, out_c):
        super().__init__()

        self.network = keras.models.Sequential([
            Conv2D(out_c, kernel_size=3),
            Conv2D(out_c, kernel_size=1, act=False),
        ])
        self.shortcut = Conv2D(out_c, kernel_size=1, act=False)
        self.relu = layers.Activation('relu')
    
    def call(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x+s)
        return x

class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)

        self.fc1 = layers.Conv2D(in_planes // ratio, 1, use_bias=False)
        self.relu1 = layers.Activation('relu')
        self.fc2 = layers.Conv2D(in_planes, 1, use_bias=False)

        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.conv1 = layers.Conv2D(1, kernel_size, padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        x = layers.concatenate([avg_out, max_out], axis=-1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Encoder(layers.Layer):
    def __init__(self, ch, pretrained=True):
        super().__init__()

        '''ResNet50'''
        backbone = resnet.ResNet50(
            include_top=False, 
            weights='imagenet' if pretrained else None,
            )
        self.layer0 = keras.models.Model(
            inputs = backbone.get_layer('conv1_pad').input, 
            outputs = backbone.get_layer('conv1_relu').output)
        self.layer1 = keras.models.Model(
            inputs = backbone.get_layer('pool1_pad').input, 
            outputs = backbone.get_layer('conv2_block3_out').output)
        self.layer2 = keras.models.Model(
            inputs = backbone.get_layer('conv3_block1_1_conv').input, 
            outputs = backbone.get_layer('conv3_block4_out').output)
        self.layer3 = keras.models.Model(
            inputs = backbone.get_layer('conv4_block1_1_conv').input, 
            outputs = backbone.get_layer('conv4_block6_out').output)
        
        '''Reduce feature channels'''
        self.c1 = Conv2D(ch)
        self.c2 = Conv2D(ch)
        self.c3 = Conv2D(ch)
        self.c4 = Conv2D(ch)

        '''Adjust input channels'''

    def call(self, x):
        '''Backbone: ResNet50'''
        x0 = x
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        c1 = self.c1(x1)
        c2 = self.c2(x2)
        c3 = self.c3(x3)
        c4 = self.c4(x4)

        return c1, c2, c3, c4

class MultiKernelDilatedConv(layers.Layer):
    def __init__(self, out_c):
        super().__init__()
        self.relu = layers.Activation('relu')

        self.c1 = Conv2D(out_c, kernel_size=1)
        self.c2 = Conv2D(out_c, kernel_size=3)
        self.c3 = Conv2D(out_c, kernel_size=7)
        self.c4 = Conv2D(out_c, kernel_size=11)
        self.s1 = Conv2D(out_c, kernel_size=1)

        self.d1 = Conv2D(out_c, kernel_size=3, dilation=1)
        self.d2 = Conv2D(out_c, kernel_size=3, dilation=3)
        self.d3 = Conv2D(out_c, kernel_size=3, dilation=7)
        self.d4 = Conv2D(out_c, kernel_size=3, dilation=11)
        self.s2 = Conv2D(out_c, kernel_size=1, act=False)
        self.s3 = Conv2D(out_c, kernel_size=1, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def call(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = layers.concatenate([x1, x2, x3, x4], axis=-1)
        x = self.s1(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = layers.concatenate([x1, x2, x3, x4], axis=-1)
        x = self.s2(x)
        s = self.s3(x0)

        x = self.relu(x+s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x

class DecoderBlock(layers.Layer):
    def __init__(self, out_c):
        super().__init__()

        self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.r1 = ResidualBlock(out_c)
        self.r2 = ResidualBlock(out_c)
    
    def call(self, x, s):
        x = self.up(x)
        x = layers.concatenate([x, s], axis=-1)
        x = self.r1(x)
        x = self.r2(x)
        return x

class MultiScaleFeatureFusion(layers.Layer):
    def __init__(self, out_c):
        super().__init__()

        self.up_2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.c1 = Conv2D(out_c)
        self.c2 = Conv2D(out_c)
        # self.c3 = Conv2D(out_c)
        self.c4 = Conv2D(out_c)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def call(self, f1, f2, f3):
        x1 = self.up_2(f1)
        x1 = self.c1(x1)
        x1 = layers.concatenate([x1, f2], axis=-1)
        x1 = self.up_2(x1)
        x1 = self.c2(x1)
        x1 = layers.concatenate([x1, f3], axis=-1)
        x1 = self.up_2(x1)
        x1 = self.c4(x1)

        x1 = x1 * self.ca(x1)
        x1 = x1 * self.sa(x1)

        return x1


def build_model(input_shape=(384, 384, 1), num_classes=1, pretrained=True):
    inputs = layers.Input(shape=input_shape)
    s = inputs
    if input_shape[-1] != 3: s = layers.Conv2D(3, 1)(inputs)
    s1, s2, s3, s4 = Encoder(96, pretrained)(s)
    x1 = MultiKernelDilatedConv(96)(s1)
    x2 = MultiKernelDilatedConv(96)(s2)
    x3 = MultiKernelDilatedConv(96)(s3)
    x4 = MultiKernelDilatedConv(96)(s4)
    d1 = DecoderBlock(96)(x4, x3)
    d2 = DecoderBlock(96)(d1, x2)
    d3 = DecoderBlock(96)(d2, x1)
    x = MultiScaleFeatureFusion(96)(d1, d2, d3)
    y = layers.Conv2D(num_classes, kernel_size=1)(x)
    outputs = layers.Activation('sigmoid' if num_classes==1 else 'softmax')(y)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == '__main__':
    input_tensor = tf.zeros((4, 384, 384, 1))
    model = build_model()
    model.summary()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
    print('done')
