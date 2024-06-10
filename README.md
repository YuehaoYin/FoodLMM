# FoodLMM: A Versatile Food Assistant using Large Multi-modal Model

<font size=5><div align='center' > 
<a href=https://arxiv.org/pdf/2312.14991>**Paper**</a> | 
<a href="https://huggingface.co/Yueha0/FoodLMM-Chat">**Model**</a> | 
[**Dataset**](#datasets)  | 
[**Training**](#training)  | 
[**Local Deployment**](#deployment) </a></div></font>

## Installation
```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Datasets

### Training Stage 1: Public Food Datasets
[VIREO Food-172](https://fvl.fudan.edu.cn/dataset/vireofood172/list.htm), 
[Recipe1M](http://wednesday.csail.mit.edu/temporal/release/), 
[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k#download-data), 
[FoodSeg103](https://xiongweiwu.github.io/foodseg103.html), 
[UECFoodPixComplete](https://mm.cs.uec.ac.jp/uecfoodpix/)

[//]: # ()
_Note: You only need to download the extracted Nutrition5k images provided in our [FoodDialogues](https://huggingface.co/datasets/Yueha0/FoodDialogues) instead of the original Nutrition5k dataset._

[//]: # ()

###  Stage 2: GPT-4 Generated Conversation Datasets
[FoodDialogues](https://huggingface.co/datasets/Yueha0/FoodDialogues), 
[FoodReasonSeg](https://huggingface.co/datasets/Yueha0/FoodReasonSeg)


## Training
### Training Data Preparation

Download them from the above links, and organize them as follows.

```
├── dataset
│   ├── FoodSeg103
│   │   ├── category_id.txt
│   │   ├── FoodReasonSeg_test.json
│   │   ├── FoodReasonSeg_train.json
│   │   └── Images
│   │       └── ...
│   │   └── ImageSets
│   │       └── ...
│   ├── UECFOODPIXCOMPLETE
│   │   └── data
│   │       ├── category.txt
│   │       ├── train9000.txt
│   │       ├── test1000.txt
│   │       └── UECFOODPIXCOMPLETE
│   │           └── train
│   │               └── ...
│   │           └── test
│   │               └── ...
│   ├── VireoFood172
│   │   └── train_id.json
│   │   └── ingre.json
│   │   └── foodlist.json
│   │   └── ready_chinese_food
│   │       └── ...
│   ├── Recipe1M
│   │   └── recipe1m_train_1488.json
│   │   └── images
│   │       └── ...
│   ├── Nutrition5k
│   │   ├── train_id.json
│   │   ├── cafe_1_id.json
│   │   ├── cafe_2_id.json
│   │   ├── dish_metadata_cafe1.csv
│   │   ├── dish_metadata_cafe2.csv
│   │   ├── FoodDialogues_train.json
│   │   ├── FoodDialogues_test.json
│   │   ├── images
│   │       └── ...
```

### Training Stage 1
To train FoodLMM, you need to download the pre-trained weights of [LISA-7B-v1-explanatory](https://huggingface.co/xinlai/LISA-7B-v1-explanatory) and [SAM ViT-H weights](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and set their paths in `train_config_Stage1.yaml`.

[//]: # (#### Training command)
```
deepspeed --master_port=XXX train_ds_Stage1.py --cfg_file=train_config_Stage1.yaml
```
The weights merging processes will be done autonomously, if you couldn't find the weights in the configed path ('./runs/FoodLMM_S1' by default), try the following commands.
```
cd ./runs/lisa-7b/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="PATH_TO_LISA" \
  --weight="PATH_TO_pytorch_model.bin" \
  --save_path="PATH_TO_SAVED_MODEL"
```
 
### Training Stage 2

[//]: # (#### Training command)
```
deepspeed --master_port=XXX train_ds_Stage2.py --cfg_file=train_config_Stage2.yaml
```

## Deployment
```
CUDA_VISIBLE_DEVICES=0 python online_demo.py --version='PATH_TO_FoodLMM_Chat'
```

## Citation 
If you find this project useful in your research, please consider citing:
```
@article{yin2023foodlmm,
  title={FoodLMM: A Versatile Food Assistant using Large Multi-modal Model},
  author={Yin, Yuehao and Qi, Huiyan and Zhu, Bin and Chen, Jingjing and Jiang, Yu-Gang and Ngo, Chong-Wah},
  journal={arXiv preprint arXiv:2312.14991},
  year={2023}
}
```

## Acknowledgement
-  This work is built upon [LISA](https://github.com/dvlab-research/LISA), and our datasets are generated from
[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k#download-data) and 
[FoodSeg103](https://xiongweiwu.github.io/foodseg103.html) using [GPT-4](https://chatgpt.com). 
