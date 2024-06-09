import random
from enum import Enum

import re
import numpy as np
import torch
import torch.distributed as dist

from utils.colors import COLORS_RGB

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if is_dist_avail_and_initialized():
            device = 'cuda:{}'.format(get_rank())
        else:
            device = torch.cuda.current_device()
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    if not output.shape == target.shape:
        print('output.shape', output.shape)
        print('target.shape', target.shape)
        output = output.reshape(target.shape)
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if is_main_process():
            print("\t".join(entries))
            print()

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        if is_main_process():
            print(" ".join(entries))
            print()

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    if is_dist_avail_and_initialized():
        device = 'cuda:{}'.format(get_rank())
    else:
        device = torch.cuda.current_device()

    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            # input_dict[k] = v.cuda(non_blocking=True)
            input_dict[k] = v.to(device, non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            # input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
            input_dict[k] = [ele.to(device, non_blocking=True) for ele in v]
    return input_dict


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def normalize_to_one(lst):
    total = sum(lst)
    if total == 0:
        return [0] * len(lst)

    normalized_lst = [x / total for x in lst]
    return normalized_lst


def is_plural(word):
    word = word.lower()
    if word.endswith("es"):
        return True
    if word.endswith("s") and not (word.endswith("us") or word.endswith("is")):
        return True
    return False


def print_in_file(content, file, flush=True, sys_print=True, **kwargs):
    if sys_print:
        print(content, **kwargs)
    print(content, file=file, flush=flush, **kwargs)


def convert_gray_to_color_mask(mask, color_idx=0):
    colored_mask = torch.zeros(size=(3, mask.shape[0], mask.shape[1]))
    temp_mask = mask > 0
    for i in range(3):
        if color_idx >= len(COLORS_RGB):
            colored_mask[i, ...] = torch.tensor(temp_mask) * torch.tensor(random.random())
        else:
            colored_mask[i, ...] = torch.tensor(temp_mask) * torch.tensor(COLORS_RGB[color_idx][i])
    colored_mask = colored_mask.permute(1, 2, 0)
    return colored_mask


def overlay_rgba_on_rgb(image, mask, alpha_channel=3):
    # 将 torch.Tensor 转为 NumPy 数组
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # 为 mask 添加 alpha 通道
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask
    mask_rgba[:, :, alpha_channel] = 128  # 设置 alpha 值，范围0-255

    # 验证图像和蒙版的形状
    if image.shape[:2] != mask_rgba.shape[:2]:
        raise ValueError("Image and mask must have the same height and width.")

    # 叠加 mask
    for c in range(0, 3):
        image[:, :, c] = image[:, :, c] * (1 - mask_rgba[:, :, alpha_channel] / 255.0) + \
                         mask_rgba[:, :, c] * (mask_rgba[:, :, alpha_channel] / 255.0)

    return image


def replace_token_with_nutrition_values(text_output, nurtition_output_dict, max_seg_num):
    text_output = replace_token_with_list(text_output, '[MASS_TOTAL]', nurtition_output_dict['total_mass_output'])
    text_output = replace_token_with_list(text_output, '[CAL_TOTAL]', nurtition_output_dict['total_calorie_output'])
    text_output = replace_token_with_list(text_output, '[FAT_TOTAL]', nurtition_output_dict['total_fat_output'])
    text_output = replace_token_with_list(text_output, '[CARB_TOTAL]', nurtition_output_dict['total_carb_output'])
    text_output = replace_token_with_list(text_output, '[PRO_TOTAL]', nurtition_output_dict['total_protein_output'])

    # for i in range(1, max_seg_num + 1):
    #     pattern =
    text_output = replace_token_with_list(text_output, r'\[MASS\d+\]', nurtition_output_dict['mass_output'], use_re=True)
    text_output = replace_token_with_list(text_output, r'\[CAL\d+\]', nurtition_output_dict['calorie_output'], use_re=True)
    text_output = replace_token_with_list(text_output, r'\[FAT\d+\]', nurtition_output_dict['fat_output'], use_re=True)
    text_output = replace_token_with_list(text_output, r'\[CARB\d+\]', nurtition_output_dict['carb_output'], use_re=True)
    text_output = replace_token_with_list(text_output, r'\[PRO\d+\]', nurtition_output_dict['protein_output'], use_re=True)
    return text_output


def replace_token_with_list(text, token_name, value_tensor, use_re=False):
    if use_re:
        sub_sents = re.split(token_name, text)
    else:
        sub_sents = text.split(token_name)

    # print('token_name:', token_name)
    # print('sub_sents', sub_sents)
    # print('value_tensor', value_tensor)
    temp_sent = ''
    for i in range(value_tensor.shape[0]):
        temp_sent += sub_sents[i]
        # temp_sent += ' '
        temp_sent += str(round(value_tensor[i, :].item(), 2))
    # temp_sent += ' '
    temp_sent += sub_sents[-1]
    return temp_sent
