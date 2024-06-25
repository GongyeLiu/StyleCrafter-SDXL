# StyleCrafter-SDXL


## 🔆 Introduction

Hi, this is an official implementation of [StyleCrafter](https://github.com/GongyeLiu/StyleCrafter) **in SDXL**
We train StyleCrafter on SDXL to further enhance its generated quality for style-guided image generation.


**TL;DR: Higher Resolution(1024×1024)!  More Visually Pleasing!**


## ⭐ Showcases

<div align="center">
<img src=.asset/teaser.png>
<p>Style-guided text-to-image results. Resolution: 1024 x 1024. (Compressed)</p>
</div>


## ⚙️ Setup

### Step 1: Install Python Environment

```bash
conda create -n style_crafter python=3.9
conda activate style_crafter

conda install cudatoolkit=11.8 cudnn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.29.1
pip install accelerate==0.31.0
pip install transformers tensorboard omegaconf opencv-python webdataset
```

### Step 2: Download checkpoints

Download StyleCrafter-SDXL checkpoints from [huggingface](https://huggingface.co/liuhuohuo/StyleCrafter-SDXL), and put them into the folder `./pretrained_ckpts/`.

After downloading and moving, the directiry structure should look like this:

```
pretrained_ckpts
├── image_encoder
│   ├── config.json
│   └── pytorch_model.bin
└── stylecrafter
    └── stylecrafter_sdxl.ckpt
```

## 💫 Inference

Run the following command to generate stylized videos.

```
python infer.py --style_dir testing_data/input_style \
  --prompts_file testing_data/prompts.txt \
  --save_dir testing_data/output \
  --scale 0.5
```

If you find unsatisfactory results, try slightly adjusting the scale value. Empirically, reduce the scale if it produces artifacts, and increase the scale if result is less stylized.


## 💥 Training

1. Prepare your own training data as [webdataset](https://github.com/webdataset/webdataset) style, or just modified dataset.py to adapted to your data as preferred.

2. launch the training bash(based on accelerate)

```bash
sh train.sh
```

## 📝 Training Details

As a reference, we train StyleCrafter-SDXL as the following steps:

* Train at resolution 512×512 for 80k steps, with batchsize=128, lr=5e-5, no noise offset;
* Train at resolution 1024×1024 for 80k steps, with batchsize=64, lr=2e-5, no noise offset;
* Train at resolution 1024×1024 for 40k steps, with batchsize=64, lr=1e-5, noise_offset=0.05;

We conduct all the training processes on 8 Nvidia A100 GPUs, which takes about a week to complete. Just approximation.

For more details(model arch, data process, etc.), please refer to our [paper](https://arxiv.org/abs/2312.00330):


## 🧰 More about StyleCrafter

**[StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter](https://arxiv.org/abs/2312.00330)**
</br>
GongyeLiu, 
Menghan Xia*, 
Yong Zhang, 
Haoxin Chen, 
Jinbo Xing, 
Xintao Wang,  
Ying Shan
Yujiu Yang*
<br>
(* corresponding authors)

<br>

**[StyleCrafter Github Repo](https://github.com/GongyeLiu/StyleCrafter)**(based on VideoCrafter)

<br>

**[StyleCrafter Homepage](https://gongyeliu.github.io/StyleCrafter.github.io/)**


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****

## 🙏 Acknowledgements
This repo is based on [diffusers](https://huggingface.co/docs/diffusers/index) and [accelerate](https://huggingface.co/docs/accelerate/index), and our training code for SDXL is largely modified from [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). We would like to thank them for their awesome contributions to the AIGC community. 

## 📭 Contact
If your have any comments or questions, feel free to contact <lgy22@mails.tsinghua.edu.cn>

## BibTex
```bibtex
@article{liu2023stylecrafter,
  title={StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter},
  author={Liu, Gongye and Xia, Menghan and Zhang, Yong and Chen, Haoxin and Xing, Jinbo and Wang, Xintao and Yang, Yujiu and Shan, Ying},
  journal={arXiv preprint arXiv:2312.00330},
  year={2023}
}
```