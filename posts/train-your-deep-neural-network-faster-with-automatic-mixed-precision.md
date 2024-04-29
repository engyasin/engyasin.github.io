<!--
.. title: Train your deep neural network faster with Automatic Mixed Precision
.. slug: train-your-deep-neural-network-faster-with-automatic-mixed-precision
.. date: 2022-09-23 16:53:19 UTC+02:00
.. tags: deep-learning, tips, pytorch
.. category: 
.. link: 
.. description: 
.. type: text
.. has_math: true
.. medium: yes
-->


*Have you been working on deep learning model with big size and wandered how to squeeze every possibility to save your time? or maybe you have the best GPU hardware but still find the speed too slow. Well, look at the bright side. This means you still have room for improvment :)*
<!--END_TEASER -->

One option for speeding up the deep learning model training was always stacking more digital circuts in optimized hardware devices like GPUs or TPUs. However, here we show additional option, namely, the adaptive changing of precision in order to save computation time while keeping the same accuracy at the same time.

The idea is simple, use FP32 when it's needed only, for example for small gradients. Otherwise, use FP16 precision when it's enough.


# Needed Hardware

Usually you may get some speed up in any hardware type, however, if your device is NVidia (Ampere, Volta or Turing) the speed up is about **3X** at best.

To know your device type, just issue the command `nvidia-smi`

# Needed Software

Most popular deep learning framework support this feature, like **Tensorflow** ,**Pytorch** and **MXNET**. Just to show-case, below an example of a network with pytorch is provided


# Example

First we need to define the network model:

```python
import torch
from torch import nn


class Model(nn.Module):

    def __init__(self,layer_1=16,layer_2=16):

        super().__init__(Model)

        self.fc1 = nn.Sequential(8,layer_1)
        self.fc2 = nn.Sequential(layer_1,layer_2)
        self.fc3 = nn.Sequential(layer_2,1)

    def forward(self):

        x = self.fc1(x)
        x = nn.functional(x)

        x = self.fc2(x)
        x = nn.functional(x)

        x = self.fc3(x)
        x = nn.functional(x)

```

Before running the training program , we initilize some dummy inputs/outputs


```python
batch_size = 512
data = [torch.randn(batch_size, 8, device="cuda") for _ in range(50)]
targets = [torch.randn(batch_size, 1, device="cuda") for _ in range(50)]

loss_fn = torch.nn.MSELoss().cuda()

net = Model()
```


Now, the training program is ran normally as follows (using **FP32** precision)

```python
opt = torch.optim.SGD(net.parameters(), lr=0.001)

epochs = 1
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() 
        # set_to_none=True here can modestly improve performance

```

If we want to use the special automatic precision, we should wrap the training with a *scaler*.
This scaler will change the precision as needed (between FP32 and FP16)

```python
use_amp = True

opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward() #instead of loss.backward
        scaler.step(opt) # instead of opt.step()
        scaler.update() # to prepare for next step
        opt.zero_grad() 
        # set_to_none=True here can modestly improve performance

```

To check the speedup, you can measure the runtime difference between the two last blocks.


*Thanks for reading!*

*You can find the original post as well as others in [my blog-post here](https://engyasin.github.io)*

# References:

1. https://developer.nvidia.com/automatic-mixed-precision
2. https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
