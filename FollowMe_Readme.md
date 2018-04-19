## Project: Deep Learning - Follow Me
### Writeup / README

[//]: # (Image References)

[image1]: ./imgs/Architecture.png
[image2]: ./imgs/Loss_graph.PNG
[image3]: ./imgs/Final_scores.PNG

### Introduction

In this project, a Fully Convolutional Neural network (FCN) is utilised to track a target object in a simulation environment. In this case, the target object is a chosen human character "hero". A FCN is able to detect where in an image an object is located, as it utilises convolutions to preserve the spatial information throughout an entire network. The word 'semantic' relates to distinctions between the meanings of different words or symbols (thanks dictionary.com). So, the FCN enables each pixel from the original image to be predicted, or segmented. In other words, the FCN enables one to assign classes to each pixel object in an image. Hence the term semantic segmentation, which is utilised in this project.

### Objective

The objective of the project is to construct a FCN and train the FCN to classify outputs into 3 classes of hero character, non-hero character and background. The performance of the classification is assessed using the Intersection over Union metric (IoU). In addition, there is a weighting applied to the average IoU (tracking the hero from close behind and far away, with no-hero being excluded). The weighted IoU is taken as the final score which must be >= 0.40 for a passing submission.

### FCN Architecture

As a fundamental illustration of FCN architecture, and the architecture utilised in this project, see the Figure below.

![image1]

The figure shows the three special techniques that FCNs utilise to go from being a Convolutional Neural Network (CNN) to a FCN, which are:
* Replacing the fully connected layer with a 1x1 convolutional layer
* Upsampling through use of transposed convolutional layers
* Adding skip connections which enable the network to use multiple resolution scales from different layers

The FCN is composed of an encoder and decoder. The encoder extracts feature information from an image, through convolution and the decoder upscales the encoder output to be the same size as the original image. The 1x1 enables the network to preserve spatial information, in 4D, as it replaces the 2D flattened tensor (fully connected layer).

Pretraining, the input image is resized to 160x160x3 (HxWxD). Each layer in the encoder halves the image size, and increases the depth of the image by 32 x layer_number. I stuck with decreasing factor of 2, as it seemed reasonable and it was easy to implement with upscaling by a factor of 2 as well. The decoder has the opposite effect of upscaling the size of the image by a factor of 2, and decreasing the depth accordingly. For specifics on each layer, see the table below. `SAME` padding and `relu` activation were utilised throughout the layers, with a `softmax` activation in the output layer.

| Layer           | Size       | Kernel size |   Strides  |  Upsample ratio  |
|-----------------|------------|-------------|------------|------------------|
| Input           | 160x160x3  |             |            |            |
| Encoder layer 1 | 80x80x32   |   3x3       |     2      |            |
| Encoder layer 2 | 40x40x64   |   3x3       |     2      |            |
| Encoder layer 3 | 20x20x128  |   3x3       |     2      |            |
| 1x1 Convolution | 20x20x256  |   1x1       |     1      |            |
| Decoder layer 1 | 40x40x128  |             |            |      2     |
| Decoder layer 2 | 80x80x64   |             |            |      2     |
| Decoder layer 3 | 160x160x32 |             |            |      2     |
| Output layer    | 3 classes  |   3x3       |     1      |            |

### Architecture with Code

The encoder is shown below which makes use of the `encoder_block` function.

```
    encoder_1 = encoder_block(inputs, filters=32, strides=2)
    encoder_2 = encoder_block(encoder_1, filters=64, strides=2)
    encoder_3 = encoder_block(encoder_2, filters=128, strides=2)
```

Each block performs depthwise separable 2D convolution and batch normalisation. Each layer applies an activation function. Simplistically, the separable convolution is different from regular convolution, as separable convolution reduces the required number of parameters, which increases the efficiency of the encoder. Batch normalisation is a technique for normalising the inputs to each layer within the network, rather than just normalising the input layer and has multiple benefits, such as faster training, higher learning rates, regularization and simplification of deeper network creation.

The 1x1 convolution is shown below, which makes use of the `conv2d_batchnorm` function.

```
    conv1 = conv2d_batchnorm(encoder_3, filters=256, kernel_size=1, strides=1)
```

The function utilises a regular 2D convolution, and applies batch normalisation. As previously mentioned, the 1x1 convolution enables the network to preserve spatial information in 4D, as it replaces the 2D flattened tensor (fully connected layer) of a CNN.

The decoder is shown below, which makes use of the `decoder_block` function.

```
    decoder_1 = decoder_block(conv1, encoder_2, filters=128)
    decoder_2 = decoder_block(decoder_1, encoder_1, filters=64)
    decoder_3 = decoder_block(decoder_2, inputs, filters=32)
```

Each block performs billinear upsampling by effectively repeating each row and column in an image by a specified factor. It also implements skip layers by concatenating two specified layers. In addition, each block implements two separable convolution layers to extract some more spatial information from prior layers.

The output layer uses a softmax activation function to classify the outputs into 3 classes of hero character, non-hero character and background.

I attempted a few architectures for this project. Initially, I utilised FCNs, each constructed of 1-layer and 2-layers in the encoder and decoder, and tested them out in the semantic segmentation lab. When these architectures were transferred to this project, they did not perform adequately i.e. the prediction accuracy was below the 0.4 requirement. Therefore, to increase the accuracy, I decided to increase the depth of the network by utilising 3-layers in both the encoder and decoder. I then tried a few 1x1 convolutional depths of 128, 256 and 512, and settled on 256 as it gave good results, but only after some adjustment of the hyper parameters.

### Hyper-parameters

The hyper-parameter selection in this project performed by trial and error, over 11 runs in total. At first, I trained on my local machine, with a Nvidia GTX 960m, with 4Gb of memory. Initially, I kept the epochs relatively low, starting at 5, and a batch size of 40. I found that increasing to a higher batch size on my local machine would deplete my GPU memory resouces and prevent training. My resulting IoU was below the requirements.  I monitored my training and validation loss values, and saw that there was room to increase the epochs, as both the indicators had further room for decay. Therefore, I increased them from 20, to 40 to 100. In between these runs, I adjusted the learning rate before retraining the network. In terms of the learning rate, I attempted the following values:
* 0.01 - although the loss graphs showed a general downward trend, this value produced peaky loss graphs - as in not a smooth decay over time as a result of too much noise in the system, and this indicated that I could attempt to decrease the learning rate.
* 0.001 - with this value, I found that I could meet the 0.4 requirements, if I trained for 100 epochs
* 0.005 - I tried a small increase, and found that I could meet the 0.4 requirements  with this value as well, for 100 epochs

Then, I trained in AWS with gave me access to a Tesla K80 with 11Gb of memory. This halved the training time for the neural network (approximately 80 seconds per epoch) and allowed me to increase the batch size to 64 without GPU memory issues. My final hyper-parameters were as follows:
* epochs = 100
* learning rate = 0.001
* batch size = 64
* steps per epoch = 64 (training images / batch_size)
* validation steps = 18 (validation images / batch_size)

The final loss curves and values are shown in the Figure below. Interestingly, I note a plateau of the validation loss until epoch 8 - and relative to the training loss, could indicate that the network was overfitting. However, after epoch 8, the validation loss decreases to a similar value to the training loss.

![image2]

### Final scoring

The snapshot of the final scoring is provided in the Figure below.

![image3]

The final score is 0.4186, which is above the requirement. I note that when following close to the hero, the classification is 100% with 539 true positives. However, the weakness appears to be detecting the hero from far away. This component consistently decreased the IoU weighted score, and seems to be the main stumbling block for the FCN. Further epochs and hyper-parameter tuning effectively increased this score by the highest ratio.

### Applicability to other objects

Can this network correctly classify other objects, such as a cat, in its present state? No. However, it could be retrained with additional/new labelled data with the correct masks of these object in order to classify other objects correctly.

### Future Enhancements

I would tackle future enhancements on two fronts - data enhancements and model enhancements, but it really depends on the criteria that is trying to be achieved. If I wanted to get a better score, I would collect more high quality data, trying to reduce bias in the collected data samples in an attempt to improve the far-away performance. Due to the flexibility of FCNs to take higher resolution images, instead of the downsampled 160x160x3, I would like to experiment with inputting higher resolution images to assess whether this increases prediction performance. 

In terms of the model enhancements, I could increase the depth of the FCN, experiment with adding more regularization, implement max-pooling, and see whether dropout has any sort of improvement, with cognizance that batch normalisation is also quite an effective regularizer. Ultimately though, I would like to take inspiration from tried-and-tested networks, and augment the pretrained networks (VGG-16, GoogLeNet as examples) and assess their performance. But as always, one would need to design and utilise a network that is fit for purpose. If the network needs to work very quickly, in real-time, then I imagine very deep networks may increase execution time. So it would become about optimising a given architecture for real time execution, using some techniques to optimise for inference, which are referred to here  (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py).

Just in terms of trying to select hyper-parameters, I would like to approach it more systematically, leveraging some Keras and scikit-learn features. I found a post that goes through some examples of Grid Search for hyper-parameters that I would like to apply to my next deep-learning task (https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

### Model submission

The model submission is included as the file `model_weights_v11`.
