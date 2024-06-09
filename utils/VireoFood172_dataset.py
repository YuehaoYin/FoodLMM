import json
import os
import random
import csv
import cv2
import torch
import random
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import utils.conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN

CATEGORY_QUESTION_LIST = [
    'What is the name of this dish?',
    'What is the name of the dish shown in the image?',
    "Can you tell me the dish's name?",
    "What dish is this?",
    "Can you tell me the name of this dish?",
    "What is the culinary name of this dish?",
    "Can you provide the name of the dish?",
    'What is the category of the dish presented in the image?',
    'Can you identify the dish displayed in the photo?',
    'Which dish is depicted in the picture?'
]
INGREDIENTS_QUESTION_LIST = [
    'Can you identify the ingredients present in this image?',
    'What ingredients are visible in this picture?',
    'Please detect the ingredients in this photo.',
    'Which food ingredients can you discern in this photo?',
    'Can you identify the ingredients from this picture?',
    'What are the ingredients of the dish depicted in the image?',
    'Can you list the components of the dish shown in the photo?',
    'What is the dish in the picture made of?'
]


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


class VireoFood172Dataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 224,
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = vision_tower

        self.DATA_DIR = os.path.join(base_image_dir, "VireoFood172")
        with open(os.path.join(self.DATA_DIR, 'train_id.json')) as f:
            train_id = json.load(f)
        self.train_id = train_id
        with open(os.path.join(self.DATA_DIR, 'ingre.json')) as f:
            ingre = json.load(f)
        self.ingre = ingre
        
        with open(os.path.join(self.DATA_DIR, 'foodlist.json')) as f:
            foodlist = json.load(f)
        self.foodlist = foodlist

    def __len__(self):
        return len(self.train_id)

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
        idx = random.randint(0, len(self.train_id) - 1)
        item = self.train_id[idx]
        image_path = os.path.join(self.DATA_DIR, 'ready_chinese_food', item.split("/",1)[1])
        ques_list = [i for i in range(2)]  #Disrupting the order of questions
        random.shuffle(ques_list)
        source = []

        for ques in range(2):
            if ques_list[ques]==0:
                data_conver_ques = dict()
                data_conver_ques['form'] = 'question'
                random_int = random.randint(0,2)
                if ques==0:
                    data_conver_ques['value'] = '<image>\n' + CATEGORY_QUESTION_LIST[random_int]
                else:
                    data_conver_ques['value'] = CATEGORY_QUESTION_LIST[random_int]
                source.append(data_conver_ques)
                
                data_conver_answ = dict()
                data_conver_answ['form'] = 'answer'
                title = self.foodlist[int(item.split("/")[1])-1]
                data_conver_answ['value'] = 'It is '+ title + '.'
                source.append(data_conver_answ)
        
            elif ques_list[ques]==1:
                data_conver_ques = dict()
                data_conver_ques['form'] = 'question'
                random_int = random.randint(0, len(INGREDIENTS_QUESTION_LIST) - 1)
                if ques==0:
                    data_conver_ques['value'] = '<image>\n' + INGREDIENTS_QUESTION_LIST[random_int]
                else:
                    data_conver_ques['value'] = INGREDIENTS_QUESTION_LIST[random_int]
                source.append(data_conver_ques)

                data_conver_answ = dict()
                data_conver_answ['form'] = 'answer'
                
                for ingre_indx in range(len(self.ingre)):
                    if self.ingre[ingre_indx].split(" ",1)[0] == item:
                        ingredients = ''
                        ingredients_list = self.ingre[ingre_indx].split(" ",1)[1].split(", ")

                        if len(ingredients_list)>1:
                            for g in range(len(ingredients_list)):
                                # if g == len(ingredients_list)-2:
                                #     ingredients = ingredients + ingredients_list[g] + ' and '
                                if g == len(ingredients_list)-1:
                                    ingredients = ingredients + ingredients_list[g] + '.'
                                else:
                                    ingredients = ingredients + ingredients_list[g] + ', '
                            data_conver_answ['value'] = 'The ingredients are ' + ingredients

                        elif len(ingredients_list)==1:
                            data_conver_answ['value'] = 'The ingredient is ' + ingredients_list[0] + "."
                        source.append(data_conver_answ)


        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess image for clip

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        conv = conversation_lib.default_conversation.copy()
        source = preprocess_multimodal(
            source,
            mm_use_im_start_end=conv.sep_style == conversation_lib.SeparatorStyle.TWO,
        )

        roles = {"question": conv.roles[0], "answer": conv.roles[1]}
        conversations = []
        if roles[source[0]["form"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["form"]]
            assert role == conv.roles[j % 2]   , f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
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
            False
        )
