# DiCleavePlus

DiCleavePlus is a Transformer-based model for human Dicer cleavage site prediction. We extend Cleavage Pattern by allowing cleavage sites to appear at any interval within the pattern. This architecture employs encoder part of Transformer as the feature extractor and implements attentional feature fusion (AFF) blocks to integrate different features.

<br>
<br>

## Requirement
The basic environment requirement is listed below:
* `python >= 3.11.10`
* `numpy >= 1.26.0`
* `pandas >= 2.2.1`
* `scikit-learn >= 1.4.2`
* `pytorch >= 2.3.0`

<br>

We provide conda environment file in `./env` directory. We recommend to build conda virtual environment with this file when using DiCleavePlus.

To establish the environment, use these codes below:

`git clone https://github.com/MGuard0303/DiCleavePlus.git/<YOUR DIRECTORY>`

`cd /<YOUR DERECTORY>`

`conda env create -f ./env/env.yaml`

Note that you need to install PyTorch by yourself because PyTorch provides different packages depending on the device (briefly, GPU version and CPU-only version). Please refer https://pytorch.org/get-started/locally/ and https://pytorch.org/get-started/previous-versions/ to install the proper PyTorch version.

<br>
<br>

## Usage

### Verify results from our paper

To verify results from our paper, please use **example.py** file.

`python example.py`
