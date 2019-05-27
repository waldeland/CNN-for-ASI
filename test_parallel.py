# Compatability Imports
from __future__ import print_function

import os

DEVICE_IDS = [0, 1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in DEVICE_IDS])

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device_str = os.environ["CUDA_VISIBLE_DEVICES"]
    device = torch.device("cuda:"+device_str)
else:
    raise Exception("No GPU detected for parallel scoring!")

# ability to perform multiprocessing
import multiprocessing
from joblib import Parallel, delayed

# use threading instead
# from joblib.pool import has_shareable_memory

NUM_CORES = multiprocessing.cpu_count()
print("Post-processing will run on {} CPU cores on your machine.".format(NUM_CORES))

from os.path import join
from data import readSEGY, get_slice
from texture_net import TextureNet
import itertools
import numpy as np
import tb_logger
from data import writeSEGY

# graphical progress bar for notebooks
from tqdm import tqdm

# Parameters
DATASET_NAME = "F3"
IM_SIZE = 65
N_CLASSES = 2
RESOLUTION = 1
# Inline, crossline, timeslice or full
SLICE = "inline"
SLICE_NUM = 339
BATCH_SIZE = 2**12
#BATCH_SIZE = 4050

# use distributed scoring
if RESOLUTION != 1:
    raise Exception("Currently we only support pixel-level scoring")

# Read 3D cube
data, data_info = readSEGY(join(DATASET_NAME, "data.segy"))

# Load trained model (run train.py to create trained
network = TextureNet(n_classes=N_CLASSES)
network.load_state_dict(torch.load(join(DATASET_NAME, "saved_model.pt")))
network.eval()

class ModelWrapper(nn.Module):
    """
    Wrap TextureNet for DataParallel to invoke classify method
    """

    def __init__(self, texture_model):
        super(ModelWrapper, self).__init__()
        self.texture_model = texture_model

    def forward(self, input):
        return self.texture_model.classify(input)

model = ModelWrapper(network)
model.eval()

print("RESOLUTION {}".format(RESOLUTION))

##########################################################################

# Log to tensorboard
logger = tb_logger.TBLogger("log", "Test")
logger.log_images(
    SLICE + "_" + str(SLICE_NUM),
    get_slice(data, data_info, SLICE, SLICE_NUM),
    cm="gray",
)

# Get half window size
window = IM_SIZE // 2
nx, ny, nz = data.shape

# generate full list of coordinates
# memory footprint of this isn't large yet, so not need to wrap as a generator
x_list = range(window, nx - window + 1)
y_list = range(window, ny - window + 1)
z_list = range(window, nz - window + 1)

print("-- generating coord list --")
# TODO: is there any way to use a generator with pyTorch data loader?
coord_list = list(itertools.product(x_list, y_list, z_list))

class MyDataset(Dataset):
    def __init__(self, data, window, coord_list):

        # main array
        self.data = data
        self.coord_list = coord_list
        self.window = window
        self.len = len(coord_list)

    def __getitem__(self, index):

        # TODO: current bottleneck - can we slice out voxels any faster
        pixel = self.coord_list[index]
        x, y, z = pixel
        small_cube = self.data[
            x - self.window : x + self.window + 1,
            y - self.window : y + self.window + 1,
            z - self.window : z + self.window + 1,
        ]

        return small_cube[np.newaxis, :, :, :], index

    def __len__(self):
        return self.len


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
else:
    print("Running on a single GPU... just one")

model.to(device)
data_torch = torch.cuda.FloatTensor(data)
my_loader = DataLoader(
    dataset=MyDataset(data_torch, window, coord_list),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# unroll full cube
indices = []
predictions = []

print("-- scoring on GPU --")

# Loop through center pixels in output cube
for (chunk, index) in tqdm(my_loader):
    input = chunk.to(device)
    output = model(input)
    # save and deal with it later on CPU
    indices += index.tolist()
    predictions += output.tolist()

print("-- aggregating results --")

classified_cube = np.zeros(data.shape)

def worker(classified_cube, ind):
    x, y, z = coord_list[ind]
    pred_class = predictions[ind][0][0][0][0]
    classified_cube[x, y, z] = pred_class

# launch workers in parallel with memory sharing ("threading" backend)
_ = Parallel(n_jobs=NUM_CORES, backend="threading")(
    delayed(worker)(classified_cube, ind) for ind in tqdm(indices)
)

print("-- writing segy --")
in_file = join(DATASET_NAME, "data.segy".format(RESOLUTION))
out_file = join(DATASET_NAME, "salt_{}.segy".format(RESOLUTION))
writeSEGY(out_file, in_file, classified_cube)

print("-- logging prediction --")
# log prediction to tensorboard
logger = tb_logger.TBLogger("log", "Test_scored")
logger.log_images(
    SLICE + "_" + str(SLICE_NUM),
    get_slice(classified_cube, data_info, SLICE, SLICE_NUM),
    cm="binary",
)
