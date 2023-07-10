
References:
1. https://arxiv.org/abs/1611.07004
2. https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
3. https://arxiv.org/abs/1411.1784
4. https://d2l.ai/chapter_generative-adversarial-networks/index.html
5. https://www.youtube.com/watch?v=banZhpreS2Y&list=PLEwK9wdS5g0onnKgvKxuUJN1Ojchl9Q9P&index=22
6. https://github.com/soumith/ganhacks
7. https://developers.google.com/machine-learning/gan/problems
8. https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
9. https://www.kaggle.com/code/kmldas/mnist-generative-adverserial-networks-in-pytorch
10. http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

### How Conditional GAN embdeddings work
There are many ways to encode and incorporate the class labels into the discriminator and generator models. A best practice involves using an embedding layer followed by a fully connected layer with a linear activation that scales the embedding to the size of the image before concatenating it in the model as an additional channel or feature map.

A version of this recommendation was described in the 2015 paper titled “Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks.”

… we also explore a class conditional version of the model, where a vector c encodes the label. This is integrated into Gk & Dk by passing it through a linear layer whose output is reshaped into a single plane feature map which is then concatenated with the 1st layer maps.

— Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks, 2015.