# Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).


### Example conda environment setup
1. Clone the repo
```bash
git clone https://github.com/WesleyHsieh0806/Amodal-Expander.git --recurse-submodules
cd Amodal-Expander
```

If you forget to add `--recurse-submodules`, do `git submodule init` and then `git submodule update`.


2. Set up conda environment
```bash
conda create --name Amodal-Expander python=3.8 -y
conda activate Amodal-Expander

bash environment_setup.sh
```