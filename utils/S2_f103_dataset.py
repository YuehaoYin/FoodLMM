import glob
import json
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import utils.conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from .OOD_nouns import OOD_NOUNS
from .colors import COLORS_NAME
from .utils import is_main_process, normalize_to_one, is_plural
from .utils import DEFAULT_IMAGE_TOKEN


def preprocess_multimodal(source, mm_use_im_start_end):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            # print(conversation_lib.default_conversation.version)
            # print('------------------')
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source


class S2F103Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
            self,
            base_image_dir,
            vision_tower,
            image_size: int = 224,
            seg_num: int = 20,
            is_val=False,
            model_name='',
    ):
        assert seg_num > 0
        self.base_image_dir = base_image_dir
        self.clip_image_processor = vision_tower
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.seg_num = seg_num
        # self.mask_colors = COLORS_NAME
        sample_nums = []
        # classes, images, masks = eval("init_{}".format('foodseg103'))(base_image_dir, is_val=False)
        # sample_nums.append(len(images))
        # self.data2list = (images, masks)
        # self.data2classes = classes
        foodseg103_data_root = os.path.join(base_image_dir, "FoodSeg103")
        with open(os.path.join(foodseg103_data_root, "category_id.txt")) as f:
            category_lines = f.readlines()
            foodseg103_classes = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
            f.close()
        self.classes = np.array(foodseg103_classes)
        self.image_root = foodseg103_data_root
        if is_val:
            with open(os.path.join(self.image_root, 'FoodReasonSeg_test.json')) as f:
                data = json.load(f)
            self.img_root = os.path.join(foodseg103_data_root, "Images", "img_dir", "test")
            self.mask_root = os.path.join(foodseg103_data_root, "Images", "ann_dir", "test")
        else:
            with open(os.path.join(self.image_root, 'FoodReasonSeg_train.json')) as f:
                data = json.load(f)
            self.img_root = os.path.join(foodseg103_data_root, "Images", "img_dir", "train")
            self.mask_root = os.path.join(foodseg103_data_root, "Images", "ann_dir", "train")
        self.data = data
        self.is_val = is_val
        self.model_name = model_name
        print('S2 dataset mdoel=', model_name)

    def __len__(self):
        return len(self.data)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if not self.is_val:
            idx = random.randint(0, len(self.data) - 1)
        item = self.data[idx]
        dish_id = item['dish_id']
        image_path = os.path.join(self.img_root, dish_id + '.jpg')
        gt_mask_path = os.path.join(self.mask_root, dish_id + '.png')

        gt_mask = Image.open(gt_mask_path)
        gt_mask = np.array(gt_mask)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # preprocess image for clip
        image_clip_embed = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        unique_categories_label = np.unique(gt_mask).tolist()

        # [yyh] remove background and others
        if 255 in unique_categories_label:
            unique_categories_label.remove(255)
        if 0 in unique_categories_label:
            unique_categories_label.remove(0)
        if 103 in unique_categories_label:
            unique_categories_label.remove(103)
        if len(unique_categories_label) == 0:
            return self.__getitem__(0)

        classes = [self.classes[class_id] for class_id in unique_categories_label]
        # print('[{}] classes:'.format(dish_id), classes)
        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        roles = {"question": conv.roles[0], "answer": conv.roles[1]}
        conversations = []
        if roles[source[0]["form"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        # print('source2', source)
        conv.messages = []
        mentioned_classes = []
        class_ids = []
        seg_token_count = 0
        for j, sentence in enumerate(source):
            if sentence['form'] == 'answer':
                seg_tokens = re.findall(r"\[SEG\d+\]", sentence['value'])
                seg_token_count += len(seg_tokens)
                mentioned_classes.extend(re.findall(r"\{(.*?)\} \[SEG\d+\]", sentence['value']))
                s_updated = re.sub(r'\{(.*?)\}', r'\1', sentence['value'])

                token_counts = {'num': 0}

                def replace_token(match):
                    token_counts['num'] += 1
                    return f"[SEG{token_counts['num']}]"

                pattern = r"\[SEG\d+\]"
                s_updated = re.sub(pattern, replace_token, s_updated)

                # 用于 LISA 模型测试
                if self.model_name == 'LISA':
                    # 正则表达式来查找所有[SEGn]实例
                    pattern = r"\[SEG\d+\]"
                    # 替换模式
                    replacement = "[SEG]"
                    # 使用re.sub来替换所有实例
                    s_updated = re.sub(pattern, replacement, s_updated)
                    # print(s_updated)
            else:
                s_updated = re.sub(r'\[SEG\d*\]', '', sentence['value'])
                s_updated = re.sub(r'\{(.*?)\}', r'\1', s_updated)
                # print('plural sentence', sentence)
            role = roles[sentence["form"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, s_updated.split(r'</s>')[0])

        for mentioned_class in mentioned_classes:
            # mentioned_class = ''.join(mentioned_class)
            try:
                class_ids.append(self.classes.tolist().index(mentioned_class.lower()))
            except:
                li = [
                    mentioned_class[:-1],
                    mentioned_class[:-3] + 'y',
                    mentioned_class[:-2],
                    mentioned_class[:-3],
                    mentioned_class[:-1].lower(),
                    str(mentioned_class[:-3] + 'y').lower(),
                    mentioned_class[:-2].lower(),
                    mentioned_class[:-3].lower(),
                ]
                # print('li', li)
                # print(self.classes)
                for word in li:
                    if word in 'duck' or word in 'chicken':
                        word = 'chicken duck'
                    if word in self.classes.tolist():
                        class_ids.append(self.classes.tolist().index(word))
                        break

        if len(class_ids) != seg_token_count:
            # print('len(mentioned_classes)', len(mentioned_classes))
            # print('len(class_ids)', len(class_ids))
            # print('seg_token_count', seg_token_count)
            return self.__getitem__(idx + 1)

        # print(conv.get_prompt())
        conversations.append(conv.get_prompt())
        questions = conversations
        sampled_classes = conversations
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        gt_mask = torch.from_numpy(gt_mask).long()
        masks = []
        for class_id in class_ids:
            masks.append(gt_mask == class_id)
        if len(masks) > 0:
            masks = torch.stack(masks, dim=0)
        else:
            masks = torch.zeros(0, *ori_size)
            gt_mask = torch.ones(ori_size) * self.ignore_label
        mass_gt = []
        calorie_gt = []
        fat_gt = []
        carb_gt = []
        protein_gt = []
        total_mass_gt = []
        total_calorie_gt = []
        total_fat_gt = []
        total_carb_gt = []
        total_protein_gt = []
        return (
            image_path,
            image,
            image_clip_embed,
            conversations,
            masks,
            gt_mask,
            resize,
            questions,
            sampled_classes,
            mass_gt,
            calorie_gt,
            fat_gt,
            carb_gt,
            protein_gt,
            total_mass_gt,
            total_calorie_gt,
            total_fat_gt,
            total_carb_gt,
            total_protein_gt,
            self.is_val,
        )
