import os
from PIL import Image
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import NN as nn
import math
from tqdm import tqdm

def load_images_flat(root_dir,
                     extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG'),
                     to_rgb = False,
                     recursive= True
                     ):
    # returns a list of numpy arrays of MNIST data
    arrays = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                path = os.path.join(dirpath, fname)
                with Image.open(path) as img:
                    img = img.convert('RGB') if to_rgb else img.convert('L')
                    arr = (np.asarray(img)/255-0.5)*2 #original image 0 to 255 with majority of 0s which is difficult to learn
                    # we change the scale from 0/255 to -1/1
                arrays.append(arr)
        if not recursive:
            break
    return(np.array(arrays))

class DataLoader:
    #loading the dataset
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle # do we shuffle the data
        self.n = len(data)
        self.idx = 0 #batch id
        self.order = np.arange(self.n)
        if self.shuffle: #shuffle
            np.random.shuffle(self.order)
    def next_batch(self):
        if self.idx + self.batch_size > self.n:
            if self.shuffle:
                np.random.shuffle(self.order)
            self.idx = 0
        inds = self.order[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        return self.data[inds]

class Discriminator:
    def __init__(self):
        self.model = nn.model([
            nn.Linear(784, 1024),
            nn.ReLU(),
            #n.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(256, 1),
            #nn.Sigmoid(), not good for numerical stability in our implementation
        ])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1) #we flatten the x for the entry
        return(self.model.forward(x))
    def backward(self, dout):
        return(self.model.backward(dout))

class Generator:
    def __init__(self):
        self.model = nn.model([
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        ])
    def forward(self, x):
        output = self.model.forward(x)
        out = output.reshape(output.shape[0], 1, 28, 28)
        return(out)
    
    def backward(self, dout):
        return(self.model.backward(dout))

def set_train_mode(m, training=True):
    # to change between eval and train
    # we deactivate the dropout out of training
    for layer in m.layers:
        if hasattr(layer, "training"):
            layer.training = training

def bce_with_logits_loss_and_grad(logits, target):
    # the loss (bce)
    # returns the loss and the gradient
    max_val = np.maximum(0.0, logits)
    loss_elem = max_val - logits * target + np.log(np.exp(-max_val) + np.exp(logits - max_val))
    loss = loss_elem.mean()
    sig = 1.0 / (1.0 + np.exp(-logits)) #sigmoid 
    dlogit = (sig - target) / np.prod(logits.shape[:-1])
    return(loss, dlogit)


def zero_grads(m):
    # reset the gradients to 0 at every epoch
    # m: model
    for layer in m.layers: #we reset every layers
        if hasattr(layer, "get_params"):
            for (_, g, _) in layer.get_params():
                g.fill(0.0)

def sample_noise(batch_size):
    return(np.random.randn(batch_size, 100).astype(np.float32))

def save_snapshot_plot(generator, epoch, batch_noise, out_dir="snapshots", n_points=64):
    samples = generator.forward(batch_noise[:n_points])   # (n,1,28,28)
    if samples.ndim == 4:
        samples = samples.squeeze(1)                      # (n,28,28)
    n = samples.shape[0]
    cols = int(min(8, math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis('off')
        if i < n:
            ax.imshow(samples[i], cmap='gray', vmin=-1.0, vmax=1.0)
    fig.suptitle(f"Epoch {epoch}", fontsize=12)
    fname = os.path.join(out_dir, f"snapshot_epoch_{epoch:05d}.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return(fname)

def make_mosaic(image_files, cols=5, figsize_per_img=2.5):
    # creates a mosaic with all the snapshot from training
    if len(image_files) == 0:
        print("No snapshots to show.")
        return
    rows = math.ceil(len(image_files) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_per_img, rows*figsize_per_img))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis('off')
    for i, fname in enumerate(image_files):
        img = plt.imread(fname)
        axes[i].imshow(img)
        axes[i].axis('off')
    fig.suptitle("Generator snapshots during training", fontsize=14)
    plt.tight_layout()
    plt.show()

def make_adam_state(model):
    # initialize the adam_state 
    state = []
    for layer in model.layers:
        if hasattr(layer, "get_params"):
            layer_state = []
            for (_, g, _) in layer.get_params():
                layer_state.append({
                    "m": np.zeros_like(g), #first moment estimate of gradients 
                    "v": np.zeros_like(g) #second moment 
                })
            state.append(layer_state)
        else:
            state.append(None) # no need to state if no param (no weights or biases to train)
            # this is for dropout, relu, sigmoid etc
    return(state)

def adam_step(model, state, lr=1e-3, beta1=0.5, beta2=0.999, eps=1e-8, t=1):
    # updates parameters
    for layer, layer_state in zip(model.layers, state):
        if layer_state is None:
            continue
        for (p, g, name), s in zip(layer.get_params(), layer_state):
            s["m"] = beta1 * s["m"] + (1 - beta1) * g
            s["v"] = beta2 * s["v"] + (1 - beta2) * (g * g)
            m_hat = s["m"] / (1 - beta1 ** t)
            v_hat = s["v"] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train(
    lrD = 1e-3,
    lrG = 1e-3,
    n_epochs = 300,
    batch_size = 32,
    print_every = 50,
    snapshot_noise_size = 1000,
    snapshots_dir = "snapshots"
):
    os.makedirs(snapshots_dir, exist_ok=True)
    D = Discriminator()
    G = Generator()
    snapshot_noise = sample_noise(snapshot_noise_size)
    saved_snapshot_files = []
    adam_state_D = make_adam_state(D.model)
    adam_state_G = make_adam_state(G.model)
    t_G, t_D = 0, 0 #adam step

    root = r"MNIST - JPG - training"
    dataset = load_images_flat(root).astype(np.float32)
    if dataset.ndim == 3:
        dataset = dataset[:, None, :, :]   #ensure size # (N,1,28,28)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    num_batches = len(dataset) // batch_size #we divide the dataset in num_batches batches
    for epoch in range(1, n_epochs + 1):
        with tqdm(total=num_batches, desc=f"Epoch {epoch}/{n_epochs}", unit="batch", leave= False) as pbar: #load bar
            for step in range(num_batches):
                real_x = dataloader.next_batch().astype(np.float32)
                set_train_mode(D.model, training=True)
                set_train_mode(G.model, training=True)
                zero_grads(D.model) #reset
                zero_grads(G.model)

                n = real_x.shape[0]
                real_y = np.ones((n, 1), dtype=np.float32) * 0.9 #better to put 0.9 than 1 

                z = sample_noise(n)
                fake_x = G.forward(z) #generate some data with G
                fake_y = np.zeros((n, 1), dtype=np.float32) #tell that they are fake

                all_x = np.vstack([real_x, fake_x]) #combine real and fake into one dataset
                all_y = np.vstack([real_y, fake_y])

                pred_all = D.forward(all_x) #we use D to predict on this combined dataset
                loss_all, grad_all = bce_with_logits_loss_and_grad(pred_all, all_y) #calculate the performance of D
                D.backward(grad_all) #backpropagate on D 

                loss_D = loss_all

                t_D += 1
                adam_step(D.model, adam_state_D, lr=lrD, t = t_D) #adam step to update

                set_train_mode(G.model, training=True) #now we train G so D is in evaluation
                set_train_mode(D.model, training=False)

                z = sample_noise(n)
                fake_x = G.forward(z) # we generate some data with G
                pred_fake_for_G = D.forward(fake_x) # and evaluate it with D
                target_for_G = np.ones_like(pred_fake_for_G, dtype=np.float32) #tell the model these are from G
                loss_G, dloss_dpred = bce_with_logits_loss_and_grad(pred_fake_for_G, target_for_G) #loss and grads

                dG_out = D.model.backward(dloss_dpred) #calculate the gradient

                G.backward(dG_out) #backprop

                t_G += 1
                adam_step(G.model,adam_state_G, lr=lrG, t = t_G)
                pbar.update(1) #update advencement bar

            if epoch % print_every == 0 or epoch == 1:
                # prints
                print(f"Epoch {epoch:4d} | loss_D={loss_D:.6f} | loss_G={loss_G:.6f} ")
                fname = save_snapshot_plot(G, epoch, snapshot_noise, out_dir=snapshots_dir, n_points=9)
                saved_snapshot_files.append(fname)

    set_train_mode(D.model, training=False)
    set_train_mode(G.model, training=False)
    make_mosaic(saved_snapshot_files, cols=4, figsize_per_img=3)

if __name__ == "__main__":
    #np.random.seed(0)
    train(
    lrD = 1e-4,
    lrG = 3e-4,
    n_epochs = 50,
    batch_size = 128,
    print_every = 5,
    snapshot_noise_size = 1000,
    snapshots_dir = "snapshots"
)
