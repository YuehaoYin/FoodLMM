import json
import os
import random

import cv2
import re
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from wandb.util import np

import utils.conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

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


class S2N5kDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 224,
        is_val: bool = False
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = vision_tower

        self.image_root = os.path.join(base_image_dir, "Nutrition5k", 'images')
        DATA_DIR = os.path.join(base_image_dir, "Nutrition5k")
        if is_val:
            with open(os.path.join(DATA_DIR, 'FoodDialogues_test.json')) as f:
                data = json.load(f)
        else:
            with open(os.path.join(DATA_DIR, 'FoodDialogues_train.json')) as f:
                data = json.load(f)
        # print(len(data))
        self.data = data
        self.is_val = is_val
        # print(len(self.data))

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

        # 自定义函数来获取并删除特殊 token 前的数字
        def extract_and_remove(match):
            value = match.group(1)
            token_type = match.group(4)  # 使用第四个匹配组
            if token_type is None:
                return match.group(0)  # 返回完整匹配的字符串
            extracted_values[token_type].append(value)
            return "[{}]".format(token_type)

        # 初始化一个字典来存储每种类型的提取值
        extracted_values = {
            "MASS": [],
            "CAL": [],
            "FAT": [],
            "CARB": [],
            "PRO": [],
            "MASS_TOTAL": [],
            "CAL_TOTAL": [],
            "FAT_TOTAL": [],
            "CARB_TOTAL": [],
            "PRO_TOTAL": [],
        }

        idx = random.randint(0, len(self.data) - 1)
        item = self.data[idx]
        imgs = os.listdir(os.path.join(self.image_root, str(item['dish_id'])))
        oh_img = os.path.join(self.image_root, str(item['dish_id']), 'rgb.png')
        if 'rgb.png' in imgs:
            imgs.remove('rgb.png')
        an_img = imgs[0]
        # print(oh_img, an_img)
        if random.random() > 0.5 and os.path.exists(oh_img):
            image_path = oh_img
        else:
            image_path = os.path.join(self.angle_images, str(item['dish_id']), an_img)

        # if os.path.exists(oh_img):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  # preprocess image for clip

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = item["conversations"]
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )
        # print('source1', source)

        roles = {"question": conv.roles[0], "answer": conv.roles[1]}
        conversations = []
        if roles[source[0]["form"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        # print('source2', source)
        conv.messages = []
        for j, sentence in enumerate(source):
            if j % 2 == 1:
                sentence["value"] = sentence["value"].replace('TOTAL_MASS', 'MASS_TOTAL')
                sentence["value"] = sentence["value"].replace('TOTAL_CAL', 'CAL_TOTAL')
                sentence["value"] = sentence["value"].replace('TOTAL_FAT', 'FAT_TOTAL')
                sentence["value"] = sentence["value"].replace('TOTAL_CARB', 'CARB_TOTAL')
                sentence["value"] = sentence["value"].replace('TOTAL_PRO', 'PRO_TOTAL')

                # 构建正则表达式模式
                tokens = "|".join(map(re.escape, extracted_values.keys()))
                pattern = rf'(\d+(\.\d+)?)\s*(grams?|g|kcal)?\s*\[({tokens})\]'
                # 使用 re.sub 修改字符串并通过自定义函数提取值
                s_updated = re.sub(pattern, extract_and_remove, sentence["value"])
                # check:
                # print(s_updated)
                # print(len(extracted_values['MASS_TOTAL']), s_updated.count("[MASS_TOTAL]"))
                # else:
                    # print('ok')

                tokens = ["MASS", "CAL", "FAT", "CARB", "PRO"]
                token_counts = {token: 0 for token in tokens}

                def replace_token(match):
                    token = match.group(1)
                    token_counts[token] += 1
                    return f"[{token}{token_counts[token]}]"

                pattern = r"\[({})\]".format("|".join(tokens))
                s_updated = re.sub(pattern, replace_token, s_updated)

                # print(s_updated)
                # print(extracted_values)
            else:
                s_updated = re.sub(r'\{(.*?)\}', r'\1', sentence['value'])
                # s_updated = sentence["value"]
            role = roles[sentence["form"]]
            # print('j={}, role={}, form={}, value={}'.format(j, role, sentence["form"], sentence['value']))
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, s_updated.split(r'</s>')[0])
        # 遍历字典，并将字符串值转换为 float
        for key, values in extracted_values.items():
            extracted_values[key] = [float(val) for val in values]

        if (len(extracted_values['MASS_TOTAL']) != conv.get_prompt().count("[MASS_TOTAL]")) or (
                len(extracted_values['CAL_TOTAL']) != conv.get_prompt().count("[CAL_TOTAL]")) or (
                len(extracted_values['FAT_TOTAL']) != conv.get_prompt().count("[FAT_TOTAL]")) or (
                len(extracted_values['CARB_TOTAL']) != conv.get_prompt().count("[CARB_TOTAL]")) or (
                len(extracted_values['PRO_TOTAL']) != conv.get_prompt().count("[PRO_TOTAL]")) or (
                len(extracted_values['MASS']) != len(re.findall(r'\[MASS\d+\]', conv.get_prompt()))) or (
                len(extracted_values['CAL']) != len(re.findall(r'\[CAL\d+\]', conv.get_prompt()))) or (
                len(extracted_values['FAT']) != len(re.findall(r'\[FAT\d+\]', conv.get_prompt()))) or (
                len(extracted_values['CARB']) != len(re.findall(r'\[CARB\d+\]', conv.get_prompt()))) or (
                len(extracted_values['PRO']) != len(re.findall(r'\[PRO\d+\]', conv.get_prompt()))):
            # print('pass')
            # print(conv.get_prompt())
            # print(extracted_values)
            # print('=============================')
            return self.__getitem__(0)

        # print(conv.get_prompt())
        conversations.append(conv.get_prompt())
        questions = conversations
        sampled_classes = conversations
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        mass_gt = extracted_values['MASS']
        calorie_gt = extracted_values['CAL']
        fat_gt = extracted_values['FAT']
        carb_gt = extracted_values['CARB']
        protein_gt = extracted_values['PRO']
        total_mass_gt = extracted_values['MASS_TOTAL']
        total_calorie_gt = extracted_values['CAL_TOTAL']
        total_fat_gt = extracted_values['FAT_TOTAL']
        total_carb_gt = extracted_values['CARB_TOTAL']
        total_protein_gt = extracted_values['PRO_TOTAL']

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
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
            self.is_val
        )
