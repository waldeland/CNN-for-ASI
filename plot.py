from __future__ import print_function
from os.path import join

import numpy as np

from data import readSEGY, get_slice

# needed for 3D projection
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

#Parameters
dataset_name = 'F3'
resolution = 1
slice = 'inline' #Inline, crossline, timeslice or full
slice_no = 339
DEBUG = False

data, data_info = readSEGY(join(dataset_name, 'data.segy'))
classified_cube, cube_info = readSEGY(join(dataset_name, 'salt_{}.segy'.format(resolution)))

# plot cube around interesting area
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.view_init(-60, 30)
cmap = plt.get_cmap("seismic")
dilation = 25; #thickness = 16
window = 25
top_left_row = 405; top_left_col = 450
print ('loading')
cube = get_slice(data, data_info, slice, slice_no, window = window)
cube = cube[top_left_row-dilation:top_left_row+dilation, :, top_left_col-dilation:top_left_col+dilation]
cube = np.rot90(cube, k=-1, axes=(0,2))
print ('loaded')
norm = plt.Normalize(data.min(), data.max())
cs = ax.voxels(np.ones_like(cube), facecolors=cmap(norm(cube)), edgecolor='white')
# fig.colorbar(cs)
print ('plotted')
plt.show()

# plot slice (larger cross-section)
surf_data = get_slice(data, data_info, slice, slice_no)
fig = plt.figure()
ax = plt.axes()
norm_surf = surf_data[:,:]
cs = ax.matshow(norm_surf, interpolation = None, cmap = cmap)
plt.colorbar(cs)
plt.show()

# plot slice (larger cross-section)
surf_data = get_slice(data, data_info, slice, slice_no)
fig = plt.figure()
ax = plt.axes()
norm_surf = surf_data[:,:]
norm_surf = norm_surf[top_left_row-dilation:top_left_row+dilation, top_left_col-dilation:top_left_col+dilation]
cs = ax.matshow(norm_surf, interpolation = None, cmap = cmap)
plt.colorbar(cs)
plt.show()

# plot scored result in cubic form
surf_pred = classified_cube[:,:]
fig = plt.figure()
ax = plt.axes(projection='3d')
cube = surf_pred[top_left_row-dilation:top_left_row+dilation, slice_no-window:slice_no+window+1, top_left_col-dilation:top_left_col+dilation]
cube = np.rot90(cube, k=-1, axes=(0,2))
norm = plt.Normalize(data.min(), data.max())
cmap_binary = plt.get_cmap("binary")
#for elev in range(0, 360, 20):
#    for azim in range(0, 360, 20):
#        ax.view_init(elev=elev, azim=azim)
cs = ax.voxels(np.ones_like(cube), facecolors=cmap_binary(cube), edgecolor='white')
#        plt.savefig("rotation_e{}_a{}.png".format(elev, azim))
# fig.colorbar(shrink=0.9)
plt.show()

