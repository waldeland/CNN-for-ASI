Code for the paper:

Tutorial: Convolutional Neural Networks for Automated Seismic Interpretation,<br />
A. U. Waldeland, A. C. Jensen, L. Gelius and A. H. S. Solberg <br />
The Leading Edge, In Press (link)

# Setup to get started
CNNs requires time-demanding computations, consider using a computer with a fast Nvidia GPU with at least 6GB ram.
- Make sure you have all the required python packages installed: <br/> 
-- torch <br/> 
-- tensorflow (GPU enabled version is not needed - we only use this package for plotting) <br />
-- tensorflow-tensorboard <br />
-- numpy<br />
-- scipy<br />
-- segyio<br />
-- matplotlib<br />
- Clone this repository<br />
- Download the demo dataset <a src=https://www.opendtect.org/osr/pmwiki.php/Main/NetherlandsOffshoreF3BlockComplete4GB>here<a/>.
- Locate the '.segy'-file, rename it to 'data.segy' and put it in the 'F3'-folder.
 
# How to use tensorboard
- Open a terminal<br />
- cd to the code-folder<br />
- run: tensorboard --out_dir='log'<br />
- Open a web-browser and go to 127.0.0.1<br />
More information canbe found here <a src=https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard>here<a/>.
  
# Usage
- train.py - used to train the CNN<br />
- train.py - used to train the CNN

# Files
In addition, it may be usefull to have a look on these files<br/>
- texture_net.py - this is where the network is defined <br/>
- batch.py - provide functionality to generate training batches <br/>
- data.py - load/save datasets with segy-format and labelled slices as images <br/>
- tensorboard.py - connects to the tensorboard functionality <br/>
- process.py - functions to apply the network to large data cubes


# Contact
Email: anders.u.waldeland@gmail.com
