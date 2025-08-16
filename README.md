Building a GAN from scratch without using torch (just using numpy).

A GAN (Generative Adversarial Network) is a duo of 2 networks, the Generator and the Discriminator that compete against each other. The generator goal is to generate some data that looks like the one from the dataset and the Discriminator objective is to tell if a picture is from the dataset or from the Discriminator. THe hard part is to reach an equilibrium between these two networks.

This repository is built to make some GAN from scratch only using numpy and to experiment myself on some examples in order to understand what methods help to have good results. 

First I took 2 examples from this article: https://realpython.com/generative-adversarial-networks/ my objective was to reproduce the results from this article to ensure that my from scratch build was correct.
The first example from the article is a model trained on generating sine waves.
you can find it in GAN_sine.py.
Here are the results: 
<img width="1200" height="600" alt="training_GAN_sine" src="https://github.com/user-attachments/assets/bf810a7c-af2d-4a88-ad0e-b36e870f883a" />
we can see the same results than in the article except that I need more epochs in order to reach a "good" result. I first thought that it was due to my numpy implementation that must have a bit of numerical instability and therefore is not optimized but no, it was a mistake in my implementation. One epoch was seeing only one batch, which is not what is meant to append during one epoch. The model is supposed to see all the dataset (see epoch//nb_epoch batches) during 1 epoch. This will be patched in the next example.

The second example is the second one from the article. Training the model on the MNIST dataset. I didn't want to use torchvision dataset (because I want to try to do everything without torch and they are formated for torch). So I found a JPG version of the dataset here: https://github.com/teavanist/MNIST-JPG.
The training loop is basicaly the same as with the sine wave example, here are the results on 50 epochs:
<img width="1536" height="754" alt="training_GAN_MNIST" src="https://github.com/user-attachments/assets/fd62e3d9-1dfa-4663-84ec-db0036f86dbb" />
One can see that the results are quite the same as in the article.
I can say that I managed to do an architecture that produce the same results as torch. There are only 2 problems, the first one of course is the fact that numpy is not coded for cuda usage, so I can only run my code on CPU and therefore it is quite slow. The second issue is some problem regarding numerical stability. It seems that calculating the sigmoid during BCE of a already sigmoided output induces some instability. This problem is not a big concern for the moment, it will, maybe, be assessed later.

Now that we know that our GAN architecture works fine, I want to try some other examples.
The first example that I tried is on a flag dataset. The dataset is composed of 254 pictures of flags of size 20px by 13ish px. This dataset is very small (254 eltms), I of course expect a lot of overfiting but this is not my goal to fix that here, I just want to try learning on a colorised dataset.
The first step is to make all the entries the same size, so I added some padding to the flag to make them 28*28px. Why this size ? because it's the one from MNIST dataset so to keep the same sizes I kepts the same dimensions. 
The first trial was to copy the code from the MNIST GAN and juste apply it to this new dataset:
Here are the results: <img width="1536" height="754" alt="training_GAN_flag_BW" src="https://github.com/user-attachments/assets/2d5cb879-dac9-4838-a9ff-2cedb07c74cb" />
We can already see a lot of things on this BW result. First, as suspected, there are a lot of overfiting. The flags that we see seem to be the exact copys (except for a strong noise) of existing flags. We can see on the picture the vietnamese flag, the flag of the Falkland Islands and some other flags that cannot be fully assessed without seeing the colors.
So it works well on this dataset, let's now make it in colour.

In order to allow the model to learn in colour I have to had 3 channels (for RGB). In BW mode the last layer of the Generator is just a 28*28=784 layer. With 3 channels it's a 28*28*3 layer (we flatten the 3 channels). Nothing else really change in the code (except the ploting part).
here are the results: <img width="1536" height="754" alt="training_GAN_flag_RGB" src="https://github.com/user-attachments/assets/10c188ef-460c-4e7c-b6bf-9b88ad7fab23" />
These results are good, we see the flags. There are still (of course) some overfiting. But there are also some other problems. The first one is the noisy aspect of the pictures. The pictures are far from being smooth. This is because of the architecture of our model. We use for the generator and the discriminator 2 MLPs (Multi Layer Perceptron) fully connected. It does the job but it only learns pixel by pixel, not some paterns like a CNN. But I want to see how far can the MLP go, so instead of changing the architecture from MLP to some CONV layers, I want to try to force the MLP to produce a "smooth" output.
In order to do that I used the TV-loss (total variation loss). Its objective is to be a way to measure high variation between close pixels. Our new loss is loss_tot = BCE + lambda_tv*TV_loss where lambda_tv is a parameter that we can tune (between 1e-3 and 1e-5 typically). An other problem that we can see on both BW and the RGB results is that the generator seems to learn only a few patterns. This is a common problem, the generator finds a sort of "local minima" where it perform well with a few pattern and does not learn any other parttern. For the moment we will no do anything about this problem and focus on the TV-loss for reducing noise.
This is the result of training the models with the TV-loss implemented:
<img width="1536" height="754" alt="training_GAN_flag_RGB_tv_loss" src="https://github.com/user-attachments/assets/4e814e38-2279-49cf-b083-0715601703a6" />
The results seem to be less noisy but it's not that flagrant. Augmenting the number of epochs (and overfit even more) does not change that. This is most likely due to the MLP structure. We will try later to use some convutional layers instead. But first let's adress the "few patterns" (single mode collapse) problem.
A solution to this known problem is to introduce a minibatch discrimination layer in the discriminator (see page 3 of https://arxiv.org/pdf/1606.03498). Its goal is to measure closeness of inputs in a minibatch. 
I implemented a new type of layer, the minibatchDiscrimination one, trained with this new layer added to the discriminator. Here are the results:
<img width="1536" height="754" alt="training_GAN_flag_RGB_tv_loss_minibatch" src="https://github.com/user-attachments/assets/e6095e45-8dfb-4d2b-8c43-f7b91d97f9f9" />
One can see that it is much better, the diversity of the flags is much better. We still have the same overfiting problem though with almost all the flags generated being existing flags. But it's not 100% from the dataset, the 9th flag on the last looks like a combinasion of the hong kong flag and the french flag. The MLP, even if it's not an optimal way to catch details in images, seems to catch details in the flags. But 
we see that the MLP cannot get rid of the noisy aspect (which is normal) even with tv-loss. What we can try, is to see if the details that the model catches are just caught because of the overfiting or if hte MLP can really catch some details.
In order to do that we need to have a way bigger dataset than the flag one (only 254). We could do some data augmentation on the flag dataset, which would make it be maybe around 1000 elements (by fliping in both direction and applying color filters). But it would still not be enough.

So I decided to use the CIFAR-10 dataset, which is a dataset of 50 000 pictures of 10 classes of animals and objects in format 32*32. 
He are the results on this dataset with the same MLP:
<img width="1536" height="754" alt="training_GAN_MLP_CIFAR_minibatch" src="https://github.com/user-attachments/assets/2ce9a406-e2ee-4af9-aa95-54af13a67c2c" />
We can see that we dont have the single mode collapse, but we see that it does not produce a recongnisable shape. We dont let it train for enough time maybe, but the training is very long.
We see that the MLP cannot catch patterns in a resonnable training time.
Let's change the architecture.

I want to switch from D: MLP, G: MLP to a D: CNN, G:conv transpose. This is called a DCGAN (Dense Convolutional Generative Adversarial Network) which is known to be good at image generation.

tbc



