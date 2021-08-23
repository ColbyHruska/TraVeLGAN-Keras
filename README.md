# TraVeLGAN-Keras
An implementation of TraVeLGAN using Keras framework

Just a week project I'm currently working on. I'm trying to implement the [TraVeLGAN: Image-to-image Translation by Transformation Vector Learning](https://arxiv.org/abs/1902.09631) paper.


## Current State (OLD)

Just finished a very rough draft of the different networks and the training process. As spected is not learning anything so is a work in progress.
Any help, criticism, ideas are very wellcome.

## Current State (NEW!)

I've detected my error(s). First, the U-Net architecture I was using was bad done. Second, I was completely missing the cycle optimization process (in my defense is not really amazingly explaned in the paper). And thrirdly, the siamese is optimized with the generator, not independently.
Does that mean it's working? Well... No. But it's getting closer.
