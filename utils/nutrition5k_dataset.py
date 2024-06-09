import json
import os
import random
import csv
import cv2
import numpy as np
import torch
import random
import torch.nn.functional as F
from transformers import CLIPImageProcessor

import utils.conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import DEFAULT_IMAGE_TOKEN

INGREDIENT_DETAIL_CALORIES = [
    "\n\t{ing_name}: About {ing_weight} grams, roughly {ing_calories} kcal",
    "\n\t{ing_name}: Approximately {ing_weight} grams, around {ing_calories} kcal",
]
INGREDIENT_DETAILS_NUTRITION = [
    "\n\t{ing_name} (about {ing_weight} grams): \n\t\tCalories: {ing_calories} kcal \n\t\tFat: {weight_fat}g \n\t\tCarbohydrates: {weight_carbo}g \n\t\tProtein: {weight_protein}g",
    # "\n\t{ing_name} (weighting about {ing_weight} grams): \n\t\tCaloric Measure: {ing_calories} kcal \n\t\tFat Mass: {weight_fat}g \n\t\tCarbohydrate Mass: {weight_carbo}g \n\t\tProtein Mass: {weight_protein}g ",
]
TOTAL_DETAILS_NUTRITION = [
    "\n\tWeight: {total_weight}g \n\tCalories: {total_calories} kcal \n\tFat: {total_fat}g \n\tCarbohydrates: {total_carbo}g \n\tProtein: {total_protein}g",
]

# 提问总卡路里
Q0 = [
    "From the image, can you estimate the food's calories?",
    "How many calories might the pictured food have?",
    "What's the caloric content of the food in the image?",
    "Can you determine the calorie count from the displayed food?",
    "What is the energy value of the pictured food?",
    "What's your estimate on the food item's calories?",
    "From the image, can you detail the food's ingredients and calorie count?",
    "Identify the food's main components from the image and provide its caloric estimate.",
    "Examine the image and provide a calorie estimate for the food.",
    "Can you deduce the food's ingredients and caloric content from the image?",
    "By observing the image, can you give a calorie estimate for the food?",
    "Analyze the dish's image. What's the likely calorie content?",
    "Can you provide the dish's caloric value from the picture?",
    "Identify the pictured food's ingredients and give a calorie count.",
    "Based on the displayed food, what might be its calorie count?",
    "Given the image, can you provide a ballpark calorie estimate for the dish?",
]
# {ing_detials} -- INGREDIENT_DETAIL_CALORIES
A0_S = [
    "From the image, the dish's ingredient is: {ing_details} \nEstimated calories: {total_calories} kcal.",
    "Based on the image, the ingredient is: {ing_details} \nApproximate calories: {total_calories} kcal.",
    "The image shows the dish with: {ing_details} \nEstimated caloric content: {total_calories} kcal.",
    "The dish in the image contains: {ing_details} \nLikely calorie count: {total_calories} kcal.",
    "Inspecting the image, the ingredient is: {ing_details} \nCalorie estimate: {total_calories} kcal.",
    "Observing the image, the dish has: {ing_details} \nApproximate calories: {total_calories} kcal.",
    "The picture reveals the dish's ingredient as: {ing_details} \nCaloric estimate: {total_calories} kcal.",
    "From the visual, the dish comprises: {ing_details} \nEstimated energy value: {total_calories} kcal.",
    "The photo indicates the dish with: {ing_details} \nCaloric content near: {total_calories} kcal.",
    "The image displays the ingredient: {ing_details} \nEstimated calorie value: {total_calories} kcal.",
    "The image suggests the dish has: {ing_details} \nCalorie content around: {total_calories} kcal.",
]
A0_M = [
    "The image shows the dish with components: {ing_details} \nTotal estimated calories: {total_calories} kcal.",
    "From the image, ingredients include: {ing_details} \nCombined calorie count: {total_calories} kcal.",
    "The image reveals the dish containing: {ing_details} \nTotal caloric value: {total_calories} kcal.",
    "The displayed dish in the image has: {ing_details} \nTotal likely calories: {total_calories} kcal.",
    "Inspecting the image, ingredients are: {ing_details} \nCalorie estimate: {total_calories} kcal.",
    "The image indicates the dish with: {ing_details} \nOverall calories: {total_calories} kcal.",
    "The picture details the dish components as: {ing_details} \nCombined calorie estimate: {total_calories} kcal.",
    "From the visual, the dish comprises: {ing_details} \nEstimated energy value: {total_calories} kcal.",
    "The photo suggests ingredients as: {ing_details} \nCaloric content near: {total_calories} kcal.",
    "The image displays the dish components: {ing_details} \nEstimated calorie value: {total_calories} kcal.",
    "The image presents the dish with elements: {ing_details} \nApproximate calorie content: {total_calories} kcal.",
]


# 提问食物中任意成分卡路里  A1 -- INGREDIENT_DETAILS_NUTRITION
Q1 = [
    "How many calories might the {ing_names} in the image have?",
    "What's the caloric content of the {ing_names} in the picture?",
    "Can you determine the calorie count of the {ing_names} from the displayed image?",
    "From the image, what might be the caloric estimate for the {ing_names}?",
    "Can you provide a calorie estimate for the {ing_names}?",
    "What's the probable calorie content of the {ing_names}?",
    "From the visual, how many calories do the {ing_names} likely have?",
    "Observing the {ing_names} in the image, can you give a caloric estimation?",
]


# 提问食物中所有成分的卡路里脂肪碳水和蛋白质（回答包含三大营养元素的总含量）
Q2 = [
    "From the image, can you detail the dish's ingredients and their nutritional values: calories, fats, carbs, and proteins?",
    "Can you break down the ingredients in the pictured dish and provide their caloric, fat, carb, and protein content?",
    "Can you list the dish's components and their associated nutritional values?",
    "Can you identify the dish's ingredients and share their calories, fats, carbs, and protein data?",
    "What are the ingredients in the displayed dish, and what are their nutritional profiles?",
    "Can you outline the main components of the pictured dish and their caloric, fat, carbohydrate, and protein content?",
    "From the visual, can you detail the dish's ingredients and their associated nutritional metrics?",
    "Can you provide the ingredients and their caloric, fat, carb, and protein values?",
    "Can you detail the dish's components and their nutritional breakdown?",
    "Can you list this dish's ingredients and their related nutritional values?"
]
# {ing_details} --- INGREDIENT_DETAILS_NUTRITION;  total_detail --- TOTAL_DETAILS_NUTRITION
A2_S = [
    "From the image, the dish's ingredient is: {ing_details} \nIts typical nutritional values are: {total_detail}",
    "The image reveals the dish's ingredient as: {ing_details} \nIts common nutritional data suggests: {total_detail}",
    "Analyzing the image, the dish contains: {ing_details} \nThe usual nutritional facts are: {total_detail}",
    "The displayed dish in the image has: {ing_details} \nIts standard nutritional values are: {total_detail}",
    "Observing the image, the ingredient is identified as: {ing_details} \nIts typical nutritional content is: {total_detail}",
    "The image indicates the dish's ingredient: {ing_details} \nAverage nutritional stats are: {total_detail}",
    "The picture details the dish's ingredient as: {ing_details} \nIts general nutritional values suggest: {total_detail}",
    "From the visual, the dish comprises: {ing_details} \nCustomary nutritional details are: {total_detail}",
    "The photo suggests the dish's ingredient as: {ing_details} \nTypical nutritional metrics are: {total_detail}",
    "The image displays the dish's single ingredient: {ing_details} \nUsual nutritional data indicates: {total_detail}",
    "Based on the provided image, the dish's ingredient is: {ing_details} \nIts conventional nutritional values are: {total_detail}",
]
A2_M = [
    "From the image, the dish's components are: {ing_details} \nTotal nutritional values for this dish are: {total_detail}",
    "The image shows the dish's ingredients as: {ing_details} \nCombined nutritional data: {total_detail}",
    "Upon analyzing the image, the dish contains: {ing_details} \nCumulative nutritional values are: {total_detail}",
    "The image presents ingredients of the dish as: {ing_details} \nThe overall nutritional values are: {total_detail}",
    "From the observation, the dish comprises: {ing_details} \nNutritional content for this dish: {total_detail}",
    "The image highlights the dish's ingredients: {ing_details} \nOverall nutritional figures are: {total_detail}",
    "The picture displays the dish's components as: {ing_details} \nThe overall nutritional values are: {total_detail}",
    "Based on the visual, the dish consists of: {ing_details} \nNutritional details for this dish are: {total_detail}",
    "The photo identifies the dish's ingredients as: {ing_details} \nNutritional metrics for this dish are: {total_detail}",
    "The image details the dish's components as: {ing_details} \nNutritional data for this dish: {total_detail}",
    "From the image provided, the dish has these components: {ing_details} \nThe dish's nutritional values are: {total_detail}",
]


# 提问食物中任意成分的卡路里脂肪碳水和蛋白质  A3 --- INGREDIENT_DETAILS_NUTRITION
Q3 = [
    "For the {ing_names} in this dish, can you detail the calorie, fat, carbohydrate, and protein content?",
    "From the image, what's the estimated nutritional profile of the {ing_names} in terms of calories, fats, carbs, and proteins?",
    "What are the nutritional values, including calories, fats, carbs, and proteins, for the {ing_names} based on the photo?",
    "Observing the dish's image, can you give a breakdown of the nutritional content for the {ing_names}?",
    "Can you provide the nutritional details of the {ing_names} from the displayed dish, focusing on calories, fats, carbs, and proteins?",
    "What are the nutritional insights, specifically calories, fats, carbs, and proteins, for the {ing_names}?",
    "Can you specify the nutritional values for the {ing_names}?",
    "What are the estimated calories, fats, carbs, and proteins for the {ing_names}?",
    "Can you determine the nutritional content of the {ing_names} in this image?",
    "Can you outline the nutritional metrics, including calories, fats, carbohydrates, and proteins, for the {ing_names}?",
]


# 提问食物总卡路里脂肪碳水和蛋白质  A4 --- TOTAL_DETAILS_NUTRITION
Q4 = [
    "Can you provide the total caloric, fat, carbohydrate, and protein content of this dish directly?",
]

# 提问食物包含最多的营养元素
Q5 = [
    "From the photo, which macronutrient is most prominent in the meal?",
    "Based on the image, which macronutrient appears most dominant in the food?",
    "Given the visual, which macronutrient seems most abundant in this dish?",
    "From the image, which among carbohydrates, proteins, and fats stands out the most?",
    "Can you identify the primary macronutrient in the dish from the image?",
    "Considering the dish's visual, which macronutrient seems to be the leading component?",
    "Reviewing the photo, which among carbohydrates, proteins, and fats is more evident?",
    "Analyzing the image, which macronutrient seems to be the most significant?",
    "From the visual cues, which macronutrient appears most concentrated in the dish?",
    "Given the dish's appearance, which of the macronutrients seems most pronounced?"
]

# INGREDIENT_DETAILS_NUTRITION, TOTAL_DETAILS_NUTRITION,
A5 = [
    "\nTherefore, the food seems richest in {macronutrient}.",
    "\nThus, the food appears to have the highest content of {macronutrient}.",
    "\nConsequently, it looks like the food is most abundant in {macronutrient}.",
    "\nAs a result, {macronutrient} seems to be the most predominant in the food.",
    "\nIt's evident that the food contains {macronutrient} in the richest amount.",
    "\nFrom the observation, the food is particularly rich in {macronutrient}.",
    "\nHence, {macronutrient} stands out as the primary component in the food.",
    "\nClearly, the food showcases a high concentration of {macronutrient}.",
    "\nAs an outcome, the food is notably dense in {macronutrient}.",
    "\nIt follows that the food has a pronounced richness in {macronutrient}.",
    "\nBy all indications, {macronutrient} is the foremost content in the food."
]

# 给食物中的三大营养元素排序
Q6 = [
    "From the image, please rank the macronutrients: carbohydrates, fats, and proteins in descending order.",
    "Can you arrange carbohydrates, fats, and proteins from most to least present?",
    "Given the dish in the picture, how would you order the prevalence of carbohydrates, fats, and proteins?",
    "Please list carbohydrates, fats, and proteins in order of prominence.",
    "Can you determine the ranking for carbohydrates, fats, and proteins?",
    "Based on the image, how would you rank the presence of carbohydrates, proteins, and fats?",
    "Referring to the picture, please prioritize the dish's macronutrients: carbohydrates, proteins, and fats.",
    "From the given photo, can you rank the abundance of the macronutrients present?",
    "Please classify the dish's content from highest to lowest among carbohydrates, fats, and proteins."
]
A6 = [
    "\nThe dominant macronutrient in the food is {macronutrient1}, followed by {macronutrient2}, and then {macronutrient3}.",
    "\nIn the food, {macronutrient1} leads in abundance, succeeded by {macronutrient2} and, to a lesser degree, {macronutrient3}.",
    "\nThe dish primarily showcases {macronutrient1}, complemented by {macronutrient2} and ending with {macronutrient3}.",
    "\nFrom the observation, the food prominently features {macronutrient1}, then {macronutrient2}, and finally {macronutrient3}.",
    "\nThe nutritional profile indicates {macronutrient1} as the major component, trailed by {macronutrient2} and, in the least, {macronutrient3}.",
    "\nPrimarily, the food is rich in {macronutrient1}, supplemented by {macronutrient2}, and has the least of {macronutrient3}.",
    "\nThe dish seems to prioritize {macronutrient1}, with a notable presence of {macronutrient2}, and a milder concentration of {macronutrient3}.",
    "\nNutritionally speaking, the food has a peak of {macronutrient1}, a descent of {macronutrient2}, and a base of {macronutrient3}.",
    "\nThe dish's nutritional composition is most abundant in {macronutrient1}, followed by {macronutrient2}, and least in {macronutrient3}.",
    "\nIn the hierarchy of macronutrients, the dish exhibits {macronutrient1} prominently, then {macronutrient2}, concluding with {macronutrient3}."
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


class Nutrition5kDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        image_size: int = 224,
        seg_num: int = 20
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.seg_num = seg_num
        self.transform = ResizeLongestSide(image_size)
        # self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.clip_image_processor = vision_tower

        self.DATA_DIR = os.path.join(base_image_dir, "Nutrition5k")
        with open(os.path.join(self.DATA_DIR, 'train_id.json')) as f:
            train_id = json.load(f)
        self.train_id = train_id
        with open(os.path.join(self.DATA_DIR, 'cafe_1_id.json')) as f:
            cafe_1_id = json.load(f)
        self.cafe_1_id = cafe_1_id
        with open(os.path.join(self.DATA_DIR, 'cafe_2_id.json')) as f:
            cafe_2_id = json.load(f)
        self.cafe_2_id = cafe_2_id
        self.cafe_1_csv_path = os.path.join(self.DATA_DIR, 'dish_metadata_cafe1.csv')
        self.cafe_2_csv_path = os.path.join(self.DATA_DIR, 'dish_metadata_cafe2.csv')

    def __len__(self):
        return len(self.train_id) * 7  # [fake len] 7 types of question

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
        csv_path = None
        if item in self.cafe_1_id:
            csv_path = self.cafe_1_csv_path
        elif item in self.cafe_2_id:
            csv_path = self.cafe_2_csv_path
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == item:
                    source = []
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

                    images = os.listdir(os.path.join(self.DATA_DIR,'images' ,row[0]))
                    if len(images) == 1 and images[0] == 'rgb.png':
                        image_path = os.path.join(self.DATA_DIR,'images' ,row[0],'rgb.png')
                    elif len(images) > 1 and 'rgb.png' in images:
                        ratio = random.random()
                        if ratio < 0.7:
                            image_path = os.path.join(self.DATA_DIR,'images' ,row[0],'rgb.png')
                        else:
                            images.remove('rgb.png')
                            image = random.randint(0,len(images)-1)
                            image_path = os.path.join(self.DATA_DIR,'images' ,row[0],images[image])
                    elif len(images) > 1 and 'rgb.png' not in images:
                        image = random.randint(0,len(images)-1)
                        image_path = os.path.join(self.DATA_DIR,'images' ,row[0],images[image])

                    max_num_ingredients = int(len(row)/7)
                    ingredient_idx_list_t = []
                    all_ing_mass = []
                    for i in range(max_num_ingredients):
                        row_id = (i+1)*7
                        if (row[row_id].lower() == 'Deprecated'.lower()) or (row[row_id] == ''):
                            return self.__getitem__(0)
                        # cal, mass
                        if float(row[row_id + 2]) > 1 and float(row[row_id + 1]) > 1:
                            row[row_id] = row[row_id][0].upper() + row[row_id][1:]
                            ingredient_idx_list_t.append(row_id)
                            all_ing_mass.append(float(row[row_id + 1]))

                    if len(ingredient_idx_list_t) < 1:
                        return self.__getitem__(0)

                    ingredient_idx_list = []
                    idx_sorted = np.argsort(all_ing_mass)[::-1]
                    idx_sorted = idx_sorted[:self.seg_num]
                    for sid in idx_sorted:
                        ingredient_idx_list.append(ingredient_idx_list_t[sid])

                    # 一共有7种问题类型
                    max_num_ingredients = len(ingredient_idx_list)
                    question_idx = random.randint(0, 6)
                    if question_idx == 0:
                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q0)-1)
                        data_conver_ques['value'] = '<image>\n' + Q0[random_ques_idx]
                        source.append(data_conver_ques)

                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        if len(ingredient_idx_list) > 1:
                            random_answ_idx = random.randint(0,len(A0_M)-1)
                            data_conver_answ['value'] = A0_M[random_answ_idx]
                        else:
                            random_answ_idx = random.randint(0, len(A0_S) - 1)
                            data_conver_answ['value'] = A0_S[random_answ_idx]
                        ing_details = ''
                        for ing_idx in range(len(ingredient_idx_list)):
                            random_patten_idx = random.randint(0, len(INGREDIENT_DETAIL_CALORIES)-1)
                            ing_detail = INGREDIENT_DETAIL_CALORIES[random_patten_idx]
                            ing_detail = ing_detail.replace('{ing_name}',row[ingredient_idx_list[ing_idx]]).replace('{ing_weight}', "[MASS{}]".format(ing_idx+1)).replace('{ing_calories}',"[CAL{}]".format(ing_idx+1))
                            ing_details = ing_details + ing_detail
                            mass_gt.append(float(row[ingredient_idx_list[ing_idx]+1]))
                            calorie_gt.append(float(row[ingredient_idx_list[ing_idx]+2]))

                        data_conver_answ['value'] = data_conver_answ['value'].replace('{ing_details}', ing_details).replace('{total_calories}','[CAL_TOTAL]')
                        total_calorie_gt.append(float(row[1]))
                        source.append(data_conver_answ)

                    elif question_idx == 1:
                        num_ingredients = random.randint(1, max_num_ingredients)
                        sampled_ingredients = random.sample(ingredient_idx_list,num_ingredients)
                        ingredients_str = ''
                        for idx in range(num_ingredients):
                            if idx == num_ingredients-2:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]] + ' and '
                            elif idx == num_ingredients-1:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]]
                            else:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]] + ', '

                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q1)-1)
                        data_conver_ques['value'] = '<image>\n' + Q1[random_ques_idx]
                        data_conver_ques['value'] = data_conver_ques['value'].replace('{ing_names}',ingredients_str)
                        source.append(data_conver_ques)

                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
        
                        ing_details = ''
                        for ing_idx in range(len(sampled_ingredients)):
                            random_patten_idx = random.randint(0,len(INGREDIENT_DETAIL_CALORIES)-1)
                            ing_detail = INGREDIENT_DETAIL_CALORIES[random_patten_idx]
                            ing_detail = ing_detail.replace('{ing_name}',row[sampled_ingredients[ing_idx]]).replace('{ing_weight}',"[MASS{}]".format(ing_idx+1)).replace('{ing_calories}',"[CAL{}]".format(ing_idx+1))
                            ing_details = ing_details + ing_detail
                            mass_gt.append(float(row[sampled_ingredients[ing_idx]+1]))
                            calorie_gt.append(float(row[sampled_ingredients[ing_idx]+2]))
                        data_conver_answ['value'] = ing_details
                        source.append(data_conver_answ)
                    
                    elif question_idx == 2:
                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q2)-1)
                        data_conver_ques['value'] = '<image>\n' + Q2[random_ques_idx]
                        source.append(data_conver_ques)
                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        ing_details = ''
                        if len(ingredient_idx_list) > 1:
                            random_answ_idx = random.randint(0,len(A2_M)-1)
                            data_conver_answ['value'] = A2_M[random_answ_idx]
                            for ing_idx in range(len(ingredient_idx_list)):
                                random_patten_idx = random.randint(0, len(INGREDIENT_DETAILS_NUTRITION) - 1)
                                ing_detail = INGREDIENT_DETAILS_NUTRITION[random_patten_idx]
                                ing_detail = ing_detail.replace('{ing_name}',
                                                                row[ingredient_idx_list[ing_idx]]).replace(
                                    '{ing_weight}', "[MASS{}]".format(ing_idx + 1)).replace('{ing_calories}',
                                                                                            "[CAL{}]".format(
                                                                                                ing_idx + 1)).replace(
                                    '{weight_fat}', "[FAT{}]".format(ing_idx + 1)).replace('{weight_carbo}',
                                                                                           "[CARB{}]".format(
                                                                                               ing_idx + 1)).replace(
                                    '{weight_protein}', "[PRO{}]".format(ing_idx + 1))
                                ing_details = ing_details + ing_detail
                                mass_gt.append(float(row[ingredient_idx_list[ing_idx] + 1]))
                                calorie_gt.append(float(row[ingredient_idx_list[ing_idx] + 2]))
                                fat_gt.append(float(row[ingredient_idx_list[ing_idx] + 3]))
                                carb_gt.append(float(row[ingredient_idx_list[ing_idx] + 4]))
                                protein_gt.append(float(row[ingredient_idx_list[ing_idx] + 5]))
                        else:
                            random_answ_idx = random.randint(0, len(A2_S) - 1)
                            data_conver_answ['value'] = A2_S[random_answ_idx]
                            ing_details = row[ingredient_idx_list[0]]

                        total_detail = TOTAL_DETAILS_NUTRITION[0].replace('{total_weight}',"[MASS_TOTAL]").replace('{total_calories}',"[CAL_TOTAL]").replace('{total_fat}',"[FAT_TOTAL]").replace('{total_carbo}',"[CARB_TOTAL]").replace('{total_protein}',"[PRO_TOTAL]")
                        total_mass_gt.append(float(row[2]))
                        total_calorie_gt.append(float(row[1]))
                        total_fat_gt.append(float(row[3]))
                        total_carb_gt.append(float(row[4]))
                        total_protein_gt.append(float(row[5]))
                        data_conver_answ['value'] = data_conver_answ['value'].replace('{ing_details}', ing_details).replace('{total_detail}',total_detail)
                        source.append(data_conver_answ)

                    elif question_idx == 3:
                        num_ingredients = random.randint(1, max_num_ingredients)
                        sampled_ingredients = random.sample(ingredient_idx_list, num_ingredients)
                        ingredients_str = ''
                        for idx in range(num_ingredients):
                            if idx == num_ingredients-2:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]] + ' and '
                            elif idx == num_ingredients-1:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]]
                            else:
                                ingredients_str = ingredients_str + row[sampled_ingredients[idx]] + ', '

                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q3)-1)
                        data_conver_ques['value'] = '<image>\n' + Q3[random_ques_idx]
                        data_conver_ques['value'] = data_conver_ques['value'].replace('{ing_names}',ingredients_str)
                        source.append(data_conver_ques)

                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        
                        ing_details = ''

                        for ing_idx in range(len(sampled_ingredients)):
                            # # [yyh] filter
                            # if float(row[ingredient_idx_list[ing_idx]+2]) < 2:
                            #     continue

                            random_patten_idx = random.randint(0,len(INGREDIENT_DETAILS_NUTRITION)-1)
                            ing_detail = INGREDIENT_DETAILS_NUTRITION[random_patten_idx]
                            ing_detail = ing_detail.replace('{ing_name}',row[sampled_ingredients[ing_idx]]).replace('{ing_weight}',"[MASS{}]".format(ing_idx+1)).replace('{ing_calories}',"[CAL{}]".format(ing_idx+1)).replace('{weight_fat}',"[FAT{}]".format(ing_idx+1)).replace('{weight_carbo}',"[CARB{}]".format(ing_idx+1)).replace('{weight_protein}',"[PRO{}]".format(ing_idx+1))
                            ing_details = ing_details + ing_detail
                            mass_gt.append(float(row[sampled_ingredients[ing_idx]+1]))
                            calorie_gt.append(float(row[sampled_ingredients[ing_idx]+2]))
                            fat_gt.append(float(row[sampled_ingredients[ing_idx]+3]))
                            carb_gt.append(float(row[sampled_ingredients[ing_idx]+4]))
                            protein_gt.append(float(row[sampled_ingredients[ing_idx]+5]))

                        data_conver_answ['value'] = ing_details
                        source.append(data_conver_answ)

                    elif question_idx == 4:
                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        data_conver_ques['value'] = '<image>\n' + Q4[0]
                        source.append(data_conver_ques)

                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        data_conver_answ['value'] = TOTAL_DETAILS_NUTRITION[0].replace('{total_weight}',"[MASS_TOTAL]").replace('{total_calories}',"[CAL_TOTAL]").replace('{total_fat}',"[FAT_TOTAL]").replace('{total_carbo}',"[CARB_TOTAL]").replace('{total_protein}',"[PRO_TOTAL]")
                        total_mass_gt.append(float(row[2]))
                        total_calorie_gt.append(float(row[1]))
                        total_fat_gt.append(float(row[3]))
                        total_carb_gt.append(float(row[4]))
                        total_protein_gt.append(float(row[5]))
                        source.append(data_conver_answ)

                    elif question_idx == 5:
                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q5)-1)
                        data_conver_ques['value'] = '<image>\n' + Q5[random_ques_idx]
                        source.append(data_conver_ques)
                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        ing_details = ''
                        if len(ingredient_idx_list) > 1:
                            random_answ_idx = random.randint(0, len(A2_M) - 1)
                            data_conver_answ['value'] = A2_M[random_answ_idx]
                            for ing_idx in range(len(ingredient_idx_list)):
                                random_patten_idx = random.randint(0, len(INGREDIENT_DETAILS_NUTRITION) - 1)
                                ing_detail = INGREDIENT_DETAILS_NUTRITION[random_patten_idx]
                                ing_detail = ing_detail.replace('{ing_name}',
                                                                row[ingredient_idx_list[ing_idx]]).replace(
                                    '{ing_weight}', "[MASS{}]".format(ing_idx + 1)).replace('{ing_calories}',
                                                                                            "[CAL{}]".format(
                                                                                                ing_idx + 1)).replace(
                                    '{weight_fat}', "[FAT{}]".format(ing_idx + 1)).replace('{weight_carbo}',
                                                                                           "[CARB{}]".format(
                                                                                               ing_idx + 1)).replace(
                                    '{weight_protein}', "[PRO{}]".format(ing_idx + 1))
                                ing_details = ing_details + ing_detail
                                mass_gt.append(float(row[ingredient_idx_list[ing_idx] + 1]))
                                calorie_gt.append(float(row[ingredient_idx_list[ing_idx] + 2]))
                                fat_gt.append(float(row[ingredient_idx_list[ing_idx] + 3]))
                                carb_gt.append(float(row[ingredient_idx_list[ing_idx] + 4]))
                                protein_gt.append(float(row[ingredient_idx_list[ing_idx] + 5]))
                        else:
                            random_answ_idx = random.randint(0, len(A2_S) - 1)
                            data_conver_answ['value'] = A2_S[random_answ_idx]
                            ing_details = row[ingredient_idx_list[0]]

                        total_detail = TOTAL_DETAILS_NUTRITION[0].replace('{total_weight}', "[MASS_TOTAL]").replace(
                            '{total_calories}', "[CAL_TOTAL]").replace('{total_fat}', "[FAT_TOTAL]").replace(
                            '{total_carbo}', "[CARB_TOTAL]").replace('{total_protein}', "[PRO_TOTAL]")
                        total_mass_gt.append(float(row[2]))
                        total_calorie_gt.append(float(row[1]))
                        total_fat_gt.append(float(row[3]))
                        total_carb_gt.append(float(row[4]))
                        total_protein_gt.append(float(row[5]))
                        data_conver_answ['value'] = data_conver_answ['value'].replace('{ing_details}',
                                                                                      ing_details).replace(
                            '{total_detail}', total_detail)

                        macronutrient_dict = dict()
                        macronutrient_dict['fats'] = float(row[3])
                        macronutrient_dict['carbohydrates'] = float(row[4])
                        macronutrient_dict['proteins'] = float(row[5])
                        macronutrient_dict = sorted(macronutrient_dict.items(), key=lambda x: x[1])
                        random_answ_idx = random.randint(0, len(A5) - 1)
                        answ_a5 = A5[random_answ_idx].replace('{macronutrient}',macronutrient_dict[-1][0])
                        data_conver_answ['value'] = data_conver_answ['value'] + answ_a5
                        source.append(data_conver_answ)

                    elif question_idx == 6:
                        data_conver_ques = dict()
                        data_conver_ques['form'] = 'question'
                        random_ques_idx = random.randint(0,len(Q6)-1)
                        data_conver_ques['value'] = '<image>\n' + Q6[random_ques_idx]
                        source.append(data_conver_ques)
                        data_conver_answ = dict()
                        data_conver_answ['form'] = 'answer'
                        ing_details = ''
                        if len(ingredient_idx_list) > 1:
                            random_answ_idx = random.randint(0, len(A2_M) - 1)
                            data_conver_answ['value'] = A2_M[random_answ_idx]
                            for ing_idx in range(len(ingredient_idx_list)):
                                random_patten_idx = random.randint(0, len(INGREDIENT_DETAILS_NUTRITION) - 1)
                                ing_detail = INGREDIENT_DETAILS_NUTRITION[random_patten_idx]
                                ing_detail = ing_detail.replace('{ing_name}',
                                                                row[ingredient_idx_list[ing_idx]]).replace(
                                    '{ing_weight}', "[MASS{}]".format(ing_idx + 1)).replace('{ing_calories}',
                                                                                            "[CAL{}]".format(
                                                                                                ing_idx + 1)).replace(
                                    '{weight_fat}', "[FAT{}]".format(ing_idx + 1)).replace('{weight_carbo}',
                                                                                           "[CARB{}]".format(
                                                                                               ing_idx + 1)).replace(
                                    '{weight_protein}', "[PRO{}]".format(ing_idx + 1))
                                ing_details = ing_details + ing_detail
                                mass_gt.append(float(row[ingredient_idx_list[ing_idx] + 1]))
                                calorie_gt.append(float(row[ingredient_idx_list[ing_idx] + 2]))
                                fat_gt.append(float(row[ingredient_idx_list[ing_idx] + 3]))
                                carb_gt.append(float(row[ingredient_idx_list[ing_idx] + 4]))
                                protein_gt.append(float(row[ingredient_idx_list[ing_idx] + 5]))
                        else:
                            random_answ_idx = random.randint(0, len(A2_S) - 1)
                            data_conver_answ['value'] = A2_S[random_answ_idx]
                            ing_details = row[ingredient_idx_list[0]]

                        total_detail = TOTAL_DETAILS_NUTRITION[0].replace('{total_weight}', "[MASS_TOTAL]").replace(
                            '{total_calories}', "[CAL_TOTAL]").replace('{total_fat}', "[FAT_TOTAL]").replace(
                            '{total_carbo}', "[CARB_TOTAL]").replace('{total_protein}', "[PRO_TOTAL]")
                        total_mass_gt.append(float(row[2]))
                        total_calorie_gt.append(float(row[1]))
                        total_fat_gt.append(float(row[3]))
                        total_carb_gt.append(float(row[4]))
                        total_protein_gt.append(float(row[5]))
                        data_conver_answ['value'] = data_conver_answ['value'].replace('{ing_details}',
                                                                                      ing_details).replace(
                            '{total_detail}', total_detail)

                        macronutrient_dict = dict()
                        macronutrient_dict['fats'] = float(row[3])
                        macronutrient_dict['carbohydrates'] = float(row[4])
                        macronutrient_dict['proteins'] = float(row[5])
                        macronutrient_dict = sorted(macronutrient_dict.items(), key=lambda x: x[1])

                        random_answ_idx = random.randint(0, len(A6) - 1)
                        answ_a6 = A6[random_answ_idx].replace('{macronutrient1}',macronutrient_dict[-1][0]).replace('{macronutrient2}',macronutrient_dict[-2][0]).replace('{macronutrient3}',macronutrient_dict[-3][0])
                        data_conver_answ['value'] = data_conver_answ['value'] + answ_a6
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
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        # print('conversations', conversations)

        # TODO
        questions = conversations
        sampled_classes = conversations

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label
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
