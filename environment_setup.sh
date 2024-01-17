conda create --name TAO-Amodal-gtr python=3.8 -y
conda activate TAO-Amodal-gtr
pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# under your working directory
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

# Clone our modified GTR
# git clone https://github.com/WesleyHsieh0806/GTR.git --recurse-submodules
cd GTR
git submodule init 
git submodule update  # to download third party libraries such as centernet
pip install -r requirements.txt