# Compatability Imports
from __future__ import print_function

import os
DEVICE_IDS = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in DEVICE_IDS])

from os.path import join
from data import readSEGY, get_slice
from texture_net import TextureNet
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np
from torch.autograd import Variable
import tb_logger
from data import writeSEGY

# graphical progress bar for notebooks
from tqdm import tqdm

#Parameters
DATASET_NAME = 'F3'
IM_SIZE = 65
DEVICES = [0]
N_CLASSES = 2
RESOLUTION = 1
SLICE = 'inline' #Inline, crossline, timeSLICE or full
SLICE_NUM = 339
BATCH_SIZE = 1

# use distributed scoring
if RESOLUTION != 1:
  raise Exception("Currently we only support pixel-level scoring")

#Read 3D cube
data, data_info = readSEGY(join(DATASET_NAME, 'data.segy'))

#Load trained model (run train.py to create trained
network = TextureNet(n_classes=N_CLASSES)
network.load_state_dict(torch.load(join(DATASET_NAME, 'saved_model.pt')))

if torch.cuda.is_available():
  # yup, apparently for data parallel models this has cuda:0... oh well
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
  raise Exception("No GPU detected for parallel scoring!")

network.eval()

# We can set the interpretation RESOLUTION to save time.f
# The interpretation is then conducted over every n-th sample and
# then resized to the full size of the input data
print ("RESOLUTION {}".format(RESOLUTION))

##########################################################################

#Log to tensorboard
logger = tb_logger.TBLogger('log', 'Test')
logger.log_images(SLICE+'_' + str(SLICE_NUM), get_slice(data, data_info, SLICE, SLICE_NUM),cm='gray')

# classified_cube = interpret(network.classify, data, data_info, 'full', None, IM_SIZE, RESOLUTION, use_gpu=use_gpu)
# model = nn.DataParallel(network.classify)

# Get half window size

window = IM_SIZE//2
nx, ny, nz = data.shape

# generate full list of coordinates
# memory footprint of this isn't large yet, so not need to wrap as a generator
x_list = range(window, nx-window+1)
y_list = range(window, ny-window+1)
z_list = range(window, nz-window+1)

print ('-- generating coord list --')
coord_list = list(itertools.product(x_list, y_list, z_list))

class MyDataset(Dataset):

  def __init__(self, data, window, coord_list):

    # main array
    self.data = data
    self.coord_list = coord_list
    self.window = window
    self.len = len(coord_list)

  def __getitem__(self, index):

    pixel = self.coord_list[index]
    x, y, z = pixel
    small_cube = self.data[x-self.window:x+self.window+1, y-self.window:y+self.window+1, z-self.window:z+self.window+1]
    #return Variable(torch.FloatTensor(small_cube[np.newaxis, :, :, :]))
    # return torch.Tensor(voxel).float()
    return torch.FloatTensor(small_cube[np.newaxis, :, :, :])

  def __len__(self):
    return self.len


classified_cube = np.zeros(data.shape)
model = network
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(network)

model.to(device)
my_loader = DataLoader(dataset=MyDataset(data, window, coord_list),
                         batch_size=BATCH_SIZE, shuffle=False)

# Loop through center pixels in output cube
for chunk in my_loader:
    input = chunk.to(device)
    output = model(input)

in_file = join(DATASET_NAME, 'data.segy'.format(RESOLUTION))
out_file = join(DATASET_NAME, 'salt_{}.segy'.format(RESOLUTION))
writeSEGY(out_file, in_file, classified_cube)

# log prediction to tensorboard
logger = tb_logger.TBLogger('log', 'Test_scored')
logger.log_images(SLICE+'_' + str(SLICE_NUM), get_slice(classified_cube, data_info, SLICE, SLICE_NUM), cm='gray')


