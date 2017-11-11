# CNN for ASI
Code for the paper:
Tutorial: Convolutional Neural Networks for Automated Seismic Interpretation,<br />
A. U. Waldeland, A. C. Jensen, L. Gelius and A. H. S. Solberg <br />
The Leading Edge, In Press (link)

### Setup to get started
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
- Download the demo dataset [here](https://www.opendtect.org/osr/pmwiki.php/Main/NetherlandsOffshoreF3BlockComplete4GB.
- Locate the '.segy'-file, rename it to 'data.segy' and put it in the 'F3'-folder.
 
### How to use tensorboard
- Open a terminal<br />
- cd to the code-folder<br />
- run: tensorboard --outdir='log'<br />
- Open a web-browser and go to localhost:6006<br />
More information can be found [here]https://www.tensorflow.org/get_started/summaries_and_tensorboard#launching_tensorboard).
  
### Usage
- train.py - Example of how the CNN is trained<br />
- test.py - Example of how CNN can be applied to predict salt in a slice or the full cube. In addition it shows how learned attributes can be extracted.<br />

### Files
In addition, it may be usefull to have a look on these files<br/>
- texture_net.py - this is where the network is defined <br/>
- batch.py - provide functionality to generate training batches with random augmentation <br/>
- data.py - load/save datasets with segy-format and labelled slices as images <br/>
- tensorboard.py - connects to the tensorboard functionality <br/>
- utils.py - some help funcitons <br/>


# Contact
Email: anders.u.waldeland@gmail.com
