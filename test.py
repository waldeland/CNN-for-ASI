# Compatability Imports
from __future__ import print_function
from os.path import join

from data import readSEGY, get_slice
from texture_net import TextureNet
import torch
import numpy as np
from torch.autograd import Variable
from utils import printProgressBar
from scipy.interpolate import interpn
import tb_logger
from utils import interpret
from data import writeSEGY

#Parameters
dataset_name = 'F3'
subsampl = 16 #We only evaluate every n-th point
im_size = 65
use_gpu = True #Switch to toggle the use of GPU or not
log_tensorboard = True

#Read 3D cube
data, data_info = readSEGY(join(dataset_name, 'data.segy'))

#Load trained model (run train.py to create trained
network = TextureNet()
network.load_state_dict(torch.load(join('F3', 'saved_model.pt')))
if use_gpu: network = network.cuda()
network.eval()

# We can set the interpretation resolution to save time.
# The interpretation is then conducted over every n-th sample and
# then resized to the full size of the input data
resolution = 16

##########################################################################
slice = 'inline' #Inline, crossline, timeslice or full
slice_no = 339
#Log to tensorboard
logger = tb_logger.TBLogger('log', 'Test')
logger.log_images(slice+'_' + str(slice_no), get_slice(data, data_info, slice, slice_no),cm='gray')

""" Plot extracted features, class probabilities and salt-predictions for slice """
#features (attributes) from layer 5
im  = interpret( network.f5, data, data_info, slice, slice_no, im_size, resolution)
logger.log_images(slice+'_' + str(slice_no)+' _f5', im)

#features from layer 4
im  = interpret( network.f4, data, data_info, slice, slice_no, im_size, resolution)
logger.log_images(slice+'_' + str(slice_no) +' _f4', im)

#Class "probabilities"
im  = interpret( network, data, data_info, slice, slice_no, im_size, resolution)
logger.log_images(slice+'_' + str(slice_no) + '_class_prob', im)

#Class
im  = interpret( network.classify, data, data_info, slice, slice_no, im_size, resolution)
logger.log_images(slice+'_' + str(slice_no) + '_class', im)


""" Make interpretation for full cube and save to SEGY file """
classified_cube  = interpret( network.classify, data, data_info, 'full', None, im_size, 32)
in_file = join(dataset_name, 'data.segy')
out_file = join(dataset_name, 'salt.segy')
writeSEGY(out_file, in_file, classified_cube)


