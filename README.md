# test-cnn
Basic CNN for workshop

New layer size:
Height = ((input_height - kernel_dim + 2 * padding) / stride) + 1
Width = ((input_width - kernel_dim + 2 * padding) / stride) + 1

Fully connected:
input_height x input_width x input_channels


Building blocks:

- Convolutional layers
- Activation function
- Pooling layers
- Fully-connected (Linear) Layers (requires flattening)

