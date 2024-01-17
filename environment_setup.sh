# conda create --name Amodal-Expander python=3.8 -y
# conda activate Amodal-Expander
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# under your working directory
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

git submodule init 
git submodule update  # to download third party libraries such as centernet
pip install -r requirements.txt