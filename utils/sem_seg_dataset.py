import glob
import os
import random

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
from utils.utils import DEFAULT_IMAGE_TOKEN

CLASS_SEG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {q_sent} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {q_sent} shown in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {q_sent} in this picture.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment the {q_sent} in this picture, please.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Identify and segment the {q_sent} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What {is_are} the {q_sent} in this picture? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "What {is_are} the {q_sent} in this picture? Please output segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{Upper_is_are} there {q_sent} in this picture? If yes, provide the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{Upper_is_are} there {q_sent} in this picture? If yes, please show with a segmentation mask.",
]

# [yyh] ingredients identification and segmentation (10)
ALL_INGREDIENTS_SEG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you identify and segment the ingredients present in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "What ingredients are visible in this picture, and can you provide their segmentation?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please detect the ingredients in this photo and highlight each with segmentation.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Show me the segmentation masks for each ingredient you recognize in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Segment and list the ingredients you spot in this picture.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Which food ingredients can you discern in this photo? Please segment them.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Highlight the ingredients in this image using segmentation masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you delineate and identify the ingredients from this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Detect and outline the ingredients in this photo with segmentation.",
    DEFAULT_IMAGE_TOKEN + "\n" + "For every ingredient you recognize in this image, please provide its segmentation.",
]

SEG_ANSWER_LIST = [
    "Sure, {a_sent}",
    "{Upper} {a_sent}",
    "Sure, in the provided image, {a_sent}",
]


def init_foodseg103(base_image_dir, is_val=False):
    foodseg103_data_root = os.path.join(base_image_dir, "FoodSeg103")
    with open(os.path.join(foodseg103_data_root, "category_id.txt")) as f:
        category_lines = f.readlines()
        foodseg103_classes = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
    foodseg103_classes = np.array(foodseg103_classes)
    if is_val:
        foodseg103_labels = sorted(
            glob.glob(
                os.path.join(foodseg103_data_root, "Images", "ann_dir", "test", "*.png")
            )
        )
    else:
        foodseg103_labels = sorted(
            glob.glob(
                os.path.join(foodseg103_data_root, "Images", "ann_dir", "train", "*.png")
            )
        )
    foodseg103_images = [
        x.replace(".png", ".jpg").replace("ann_dir", "img_dir")
        for x in foodseg103_labels
    ]

    if is_main_process():
        print("foodseg103[{}]: ".format('test' if is_val else 'train'), len(foodseg103_images))
    return foodseg103_classes, foodseg103_images, foodseg103_labels


def init_uecfoodpix(base_image_dir, is_val=False):
    uecfoodpix_data_root = os.path.join(base_image_dir, "UECFOODPIXCOMPLETE", "data")
    with open(os.path.join(uecfoodpix_data_root, "category.txt")) as f:
        category_lines = f.readlines()
        uecfoodpix_classes = [' '.join(line_data.split('\t')[1:]).strip() for line_data in category_lines]
        f.close()
    uecfoodpix_classes = np.array(uecfoodpix_classes)
    if is_val:
        test_img_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "test", "img", "{id}.jpg")
        test_mask_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "test", "mask", "{id}.png")
        with open(os.path.join(uecfoodpix_data_root, "test1000.txt")) as f:
            id_lines = f.readlines()
            uecfoodpix_images = [test_img_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            uecfoodpix_labels = [test_mask_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            f.close()
    else:
        train_img_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "train", "img", "{id}.jpg")
        train_mask_template = os.path.join(uecfoodpix_data_root, "UECFoodPIXCOMPLETE", "train", "mask", "{id}.png")
        with open(os.path.join(uecfoodpix_data_root, "train9000.txt")) as f:
            id_lines = f.readlines()
            uecfoodpix_images = [train_img_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            uecfoodpix_labels = [train_mask_template.format(id=id_line.split('\n')[0]) for id_line in id_lines]
            f.close()
    if is_main_process():
        print("uecfoodpix[{}]: ".format('test' if is_val else 'train'), len(uecfoodpix_images))
    return uecfoodpix_classes, uecfoodpix_images, uecfoodpix_labels


class SemSegDataset(torch.utils.data.Dataset):
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
            OOD_sample_ratio: float = 0.1,
            num_OOD_per_sample: int = 0,
            prob_OOD=None,
            sem_seg_data="foodseg103||uecfoodpix",
            sem_seg_data_ratio=None,
            is_val=False,
    ):
        assert seg_num + num_OOD_per_sample > 0
        self.base_image_dir = base_image_dir
        self.clip_image_processor = vision_tower
        self.image_size = image_size
        self.transform = ResizeLongestSide(image_size)
        self.seg_num = seg_num
        self.num_OOD_per_sample = num_OOD_per_sample
        self.OOD_sample_ratio = OOD_sample_ratio
        if prob_OOD is None:
            prob_OOD = [0.8, 0.1, 0.05, 0.05]
            while len(prob_OOD) < num_OOD_per_sample:
                prob_OOD.extend(0.01)
        self.prob_OOD = prob_OOD
        self.sem_seg_datas = sem_seg_data.split("||")

        self.class_seg_question_list = CLASS_SEG_QUESTION_LIST
        self.all_ingredients_seg_question_list = ALL_INGREDIENTS_SEG_QUESTION_LIST
        self.answer_list = SEG_ANSWER_LIST
        # self.mask_colors = COLORS_NAME
        self.data2list = {}
        self.data2classes = {}
        sample_nums = []
        for ds in self.sem_seg_datas:
            classes, images, labels = eval("init_{}".format(ds))(base_image_dir, is_val=is_val)
            sample_nums.append(len(images))
            self.data2list[ds] = (images, labels)
            self.data2classes[ds] = classes
        if sem_seg_data_ratio is None:
            sem_seg_data_ratio = sample_nums
        sem_seg_data_ratio = sem_seg_data_ratio[:len(self.sem_seg_datas)]
        self.sem_seg_data_ratio = normalize_to_one(sem_seg_data_ratio)
        if is_main_process() and not is_val:
            print('[SEG] datasets: {}, ratios: {}.'.format(self.sem_seg_datas, self.sem_seg_data_ratio))
        self.is_val = is_val

    def __len__(self):
        num = 0
        for dataset in self.sem_seg_datas:
            num += len(self.data2list[dataset][0])
        return num

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        x = (x - SemSegDataset.pixel_mean) / SemSegDataset.pixel_std
        h, w = x.shape[-2:]
        padh = SemSegDataset.img_size - h
        padw = SemSegDataset.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = np.random.choice(
            range(len(self.sem_seg_datas)), size=1, p=self.sem_seg_data_ratio[:len(self.sem_seg_datas)]
        )[0]
        ds = self.sem_seg_datas[ds]

        images, gt_masks = self.data2list[ds]
        if not self.is_val:
            idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        gt_mask_path = gt_masks[idx]
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
        if ds in ["foodseg103"]:
            if 255 in unique_categories_label:
                unique_categories_label.remove(255)
            if 0 in unique_categories_label:
                unique_categories_label.remove(0)
            if 103 in unique_categories_label:
                unique_categories_label.remove(103)
            if len(unique_categories_label) == 0:
                return self.__getitem__(0)
        elif ds in ["uecfoodpix"]:
            if 255 in unique_categories_label:
                unique_categories_label.remove(255)
            if 0 in unique_categories_label:
                unique_categories_label.remove(0)
            if 101 in unique_categories_label:
                unique_categories_label.remove(101)
            if len(unique_categories_label) == 0:
                return self.__getitem__(0)
            gt_mask = gt_mask[..., 0]
        classes = [self.data2classes[ds][class_id] for class_id in unique_categories_label]
        if len(classes) == 0:
            return self.__getitem__(0)
        questions = []
        answers = []
        class_ids = []
        if len(classes) >= self.seg_num:
            sampled_classes = np.random.choice(
                classes, size=self.seg_num, replace=False
            ).tolist()
        else:
            sampled_classes = classes

        if not self.is_val:
            questions, answers, class_ids = self.generate_train_resource(sampled_classes, questions, answers, ds, class_ids, classes)

            # print(image_path, '--> sampled_classes:', sampled_classes, 'classes', classes, 'class_ids', class_ids)
        else:
            if self.seg_num > 0:
                questions, answers, class_ids = SemSegDataset.generate_test_resource_seg(
                    all_classes=self.data2classes[ds].tolist(),
                    question_list=self.class_seg_question_list,
                    answer_list=self.answer_list,
                    sampled_classes=sampled_classes,
                    questions=questions,
                    answers=answers,
                    class_ids=class_ids)
            else:
                questions, answers, class_ids = SemSegDataset.generate_test_resource_ood(
                    all_classes=self.data2classes[ds].tolist(),
                    question_list=self.class_seg_question_list,
                    answer_list=self.answer_list,
                    sampled_classes=sampled_classes,
                    questions=questions,
                    answers=answers,
                    class_ids=class_ids,
                    classes=classes)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

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

    def generate_train_resource(self, sampled_classes, questions, answers, ds, class_ids, classes):
        max_num_classes = len(sampled_classes)
        # [yyh] 1. all_ingredients
        if (random.random() > 0.6) and (ds in ["foodseg103"]):
            questions, answers, class_ids = self.generate_all_ingredients_seg_question_answer_pair(sampled_classes,
                                                                                                   questions, answers,
                                                                                                   ds,
                                                                                                   class_ids)
        else:
            # [yyh] 2. class_seg
            OOD_classes = [cls for cls in self.data2classes[ds] if (cls not in classes) and ('background' not in cls and 'other' not in cls)]
            OOD_extra = np.random.choice(OOD_NOUNS, size=int(len(OOD_classes) * 2), replace=False).tolist()
            OOD_classes.extend(OOD_extra)

            ood_ratio = random.random()
            if (self.num_OOD_per_sample > 0) and (ood_ratio < self.OOD_sample_ratio):
                num_classes = random.randint(0, max_num_classes)
                if num_classes == 0:
                    self.prob_OOD[0] = 0
            else:
                num_classes = random.randint(1, max_num_classes)
            query_classes = np.random.choice(sampled_classes, size=num_classes, replace=False).tolist()

            sampled_OOD_classes = []
            if (self.num_OOD_per_sample > 0) and (ood_ratio < self.OOD_sample_ratio):
                if num_classes == 0:
                    self.prob_OOD[0] = 0
                num_OOD_classes = self.num_OOD_per_sample + 1
                num_OOD_classes = np.random.choice(range(num_OOD_classes), size=1,
                                                   p=normalize_to_one(self.prob_OOD[:num_OOD_classes]))
                sampled_OOD_classes = np.random.choice(OOD_classes, size=num_OOD_classes, replace=False).tolist()

            questions, answers, class_ids = self.generate_class_seg_question_answer_pair(
                all_classes=self.data2classes[ds].tolist(),
                sampled_classes=query_classes,
                class_seg_question_list=self.class_seg_question_list,
                answer_list=self.answer_list,
                sampled_OOD_classes=sampled_OOD_classes,
                questions=questions,
                answers=answers,
                class_ids=class_ids)
        return questions, answers, class_ids

    def generate_all_ingredients_seg_question_answer_pair(self, sampled_classes, questions, answers, ds, class_ids):
        # [yyh]
        p_scale = len(sampled_classes)
        questions.append(random.choice(self.all_ingredients_seg_question_list))
        answer_template = random.choice(self.answer_list)
        random.shuffle(sampled_classes)
        if '{Upper}' in answer_template:
            answer_template = answer_template.replace('{Upper}', 'The')
            a_sent = ''
        else:
            a_sent = 'the '
        a_sent += sampled_classes[0]
        a_sent += ' are ' if +is_plural(sampled_classes[0]) else ' is '
        # a_sent += str('masked as [SEG] ' + COLORS_NAME[0] + ', ')
        a_sent += str('masked as [SEG1], ')
        class_ids.append(self.data2classes[ds].tolist().index(sampled_classes[0]))
        for i in range(1, p_scale - 1):
            # a_sent += str(sampled_classes[i] + ' as [SEG] ' + COLORS_NAME[i] + ', ')
            a_sent += str(sampled_classes[i] + ' as [SEG{}], '.format(i + 1))
            class_ids.append(self.data2classes[ds].tolist().index(sampled_classes[i]))
        if p_scale > 1:
            if a_sent.endswith(', '):
                a_sent = a_sent[:-2]
            a_sent += str(
                # ' and {} as [SEG] '.format(sampled_classes[p_scale - 1]) + COLORS_NAME[p_scale - 1] + '. ')
                ' and {} as [SEG{}]'.format(sampled_classes[p_scale - 1], p_scale) + '. ')
            class_ids.append(self.data2classes[ds].tolist().index(sampled_classes[p_scale - 1]))
        else:
            a_sent = a_sent[:-2]
            a_sent += '.'
        answers.append(answer_template.format(a_sent=a_sent))
        return questions, answers, class_ids

    @staticmethod
    def generate_test_resource_seg(all_classes, question_list, answer_list, sampled_classes, questions, answers, class_ids, refer=1):
        if len(sampled_classes) > refer:
            sampled_classes = np.random.choice(sampled_classes, size=refer, replace=False).tolist()
        questions, answers, class_ids = SemSegDataset.generate_class_seg_question_answer_pair(
            all_classes=all_classes,
            sampled_classes=sampled_classes,
            class_seg_question_list=question_list,
            answer_list=answer_list,
            sampled_OOD_classes=[],
            questions=questions,
            answers=answers,
            class_ids=class_ids)
        return questions, answers, class_ids

    @staticmethod
    def generate_test_resource_ood(all_classes, question_list, answer_list, sampled_classes, questions, answers,
                                   class_ids, classes, refer=1):
        # [yyh] 2. class_seg
        OOD_classes = [cls for cls in all_classes if
                       (cls not in classes) and ('background' not in cls and 'other' not in cls)]
        OOD_extra = np.random.choice(OOD_NOUNS, size=int(len(OOD_classes) * 2), replace=False).tolist()
        OOD_classes.extend(OOD_extra)
        sampled_OOD_classes = np.random.choice(OOD_classes, size=refer, replace=False).tolist()
        questions, answers, class_ids = SemSegDataset.generate_class_seg_question_answer_pair(
            all_classes=all_classes,
            sampled_classes=sampled_classes,
            class_seg_question_list=question_list,
            answer_list=answer_list,
            sampled_OOD_classes=sampled_OOD_classes,
            questions=questions,
            answers=answers,
            class_ids=class_ids)
        return questions, answers, class_ids

    @staticmethod
    def generate_class_seg_question_answer_pair(all_classes, sampled_classes, class_seg_question_list, answer_list,
                                                sampled_OOD_classes, questions, answers,
                                                class_ids):
        # [yyh]
        p_scale, OOD_scale = len(sampled_classes), len(sampled_OOD_classes)
        scale = p_scale + OOD_scale
        quest_classes = [*sampled_classes, *sampled_OOD_classes]
        random.shuffle(quest_classes)
        question_template = random.choice(class_seg_question_list)
        answer_template = random.choice(answer_list)

        if '{is_are}' in question_template:
            if scale > 1:
                question_template = question_template.replace('{is_are}', 'are')
            else:
                question_template = question_template.replace('{is_are}', 'is')
        if '{Upper_is_are}' in question_template:
            if scale > 1:
                question_template = question_template.replace('{Upper_is_are}', 'Are')
            else:
                question_template = question_template.replace('{Upper_is_are}', 'Is')
        if scale > 1:
            q_sent = str(quest_classes[0] + ', ')
        else:
            q_sent = str(quest_classes[0])
        for i in range(1, scale - 1):
            q_sent += str(quest_classes[i] + ', ')
        if scale > 1:
            if q_sent.endswith(', '):
                q_sent = q_sent[:-2]
            q_sent += ' and {}'.format(quest_classes[scale - 1])

        temp_sampled_classes = []
        temp_OOD_classes = []
        for item in quest_classes:
            if item in sampled_classes:
                temp_sampled_classes.append(item)
            else:
                temp_OOD_classes.append(item)
        sampled_classes = temp_sampled_classes
        sampled_OOD_classes = temp_OOD_classes
        if '{Upper}' in answer_template:
            answer_template = answer_template.replace('{Upper}', 'The')
            a_sent = ''
        else:
            a_sent = 'the '

        if p_scale > 0:
            a_sent += sampled_classes[0]
            a_sent += ' are ' if is_plural(sampled_classes[0]) else ' is '
            # a_sent += str('masked as [SEG] ' + COLORS_NAME[0] + ', ')
            a_sent += str('masked as [SEG1], ')
            class_ids.append(all_classes.index(sampled_classes[0]))
            for i in range(1, p_scale - 1):
                # a_sent += str(sampled_classes[i] + ' as [SEG] ' + COLORS_NAME[i] + ', ')
                a_sent += str(sampled_classes[i] + ' as [SEG{}], '.format(i + 1))
                class_ids.append(all_classes.index(sampled_classes[i]))
            if p_scale > 1:
                if a_sent.endswith(', '):
                    a_sent = a_sent[:-2]
                a_sent += str(
                    # ' and {} as [SEG] '.format(sampled_classes[p_scale - 1]) + COLORS_NAME[p_scale - 1] + '.')
                    ' and {} as [SEG{}]'.format(sampled_classes[p_scale - 1], p_scale) + '.')
                class_ids.append(all_classes.index(sampled_classes[p_scale - 1]))
            else:
                a_sent = a_sent[:-2]
                a_sent += '.'
        else:
            a_sent = ''

        if OOD_scale > 0:
            if a_sent.endswith('.'):
                a_sent += ' The '
            if OOD_scale > 1:
                for i in range(0, OOD_scale - 1):
                    a_sent += str(sampled_OOD_classes[i] + ', ')
                if a_sent.endswith(', '):
                    a_sent = a_sent[:-2]
                a_sent += str(' and ' + sampled_OOD_classes[OOD_scale - 1] + ' are not found in this picture.')
            else:
                if len(a_sent) == 0 and 'The' not in answer_template:
                    a_sent += 'The '
                a_sent += sampled_OOD_classes[0]
                a_sent += ' are ' if is_plural(sampled_OOD_classes[0]) else ' is '
                a_sent += 'not found in this picture.'

        questions.append(question_template.format(q_sent=q_sent))
        answers.append(answer_template.format(a_sent=a_sent))

        return questions, answers, class_ids
