# CNN-for-ASI
Code for the paper:

Tutorial: Convolutional Neural Networks for Automated Seismic Interpretation,<br />
A. U. Waldeland, A. C. Jensen, L. Gelius and A. H. S. Solberg <br />
The Leading Edge, In Press

# Setup
Make sure you have all the required python packages installed. Copy this repository An example dataset can be downloaded from https://www.opendtect.org/osr/pmwiki.php/Main/NetherlandsOffshoreF3BlockComplete4GB. Locate the '...sgy'-file and put it in the data-folder. Some labeled slice for this dataset are provided in this repository. <br/>
tensorboard....<br/>
train.py<br/>
test.py

# Files
In addition, it may be usefull to have a look on these files
texture_net.py - this is where the network is defined <br/>
batch.py - provide functionality to generate training batches <br/>
data.py - load/save datasets with segy-format and labelled slices as images <br/>
tensorboard.py - connects to the tensorboard functionality <br/>
process.py - functions to apply the network to large data cubes

# Dependencies
Python 2.7 with the following packages:<br />
    tensorflow (GPU enabled version is highly recommended. <br />
    numpy<br />
    scipy<br />
    segyio https://github.com/Statoil/segyio<br />
    
# Other
A similar code example can be found at https://github.com/bolgebrygg/MalenoV.

# Contact
Email: anders.u.waldeland@gmail.com
