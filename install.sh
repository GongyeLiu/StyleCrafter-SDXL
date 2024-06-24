conda create -n diffusers python=3.9
conda activate diffusers

conda install cudatoolkit=11.8 cudnn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.29.1
pip install accelerate==0.31.0
pip install transformers tensorboard omegaconf opencv-python webdataset
