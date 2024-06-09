import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from model.LISA import LISAForCausalLM
from utils.config import Config
from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="merge lora weights and save model with hf format"
    )
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--local_rank", default=-1, help="path to configuration file.")
    parser.add_argument("--weight", default="", type=str, required=True)
    parser.add_argument("--save_path", default="./runs", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format.",
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    configs = Config(args)
    # configs.pretty_print_system()
    args = configs.args
    args = argparse.Namespace(**args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.init_ckpt_dir,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    added_tokens = tokenizer.add_tokens("[MASS_TOTAL]")
    args.mass_token_idx = tokenizer("[MASS_TOTAL]", add_special_tokens=False).input_ids[0]
    added_tokens = tokenizer.add_tokens("[CAL_TOTAL]")
    args.calorie_token_idx = tokenizer("[CAL_TOTAL]", add_special_tokens=False).input_ids[0]
    added_tokens = tokenizer.add_tokens("[FAT_TOTAL]")
    args.fat_token_idx = tokenizer("[FAT_TOTAL]", add_special_tokens=False).input_ids[0]
    added_tokens = tokenizer.add_tokens("[CARB_TOTAL]")
    args.carbohydrate_token_idx = tokenizer("[CARB_TOTAL]", add_special_tokens=False).input_ids[0]
    added_tokens = tokenizer.add_tokens("[PRO_TOTAL]")
    args.protein_token_idx = tokenizer("[PRO_TOTAL]", add_special_tokens=False).input_ids[0]
    model_args = {
        "train_mask_decoder": True,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "max_seg_num": args.max_seg_num,
        "seg_token_idx": args.seg_token_idx,
        "mass_token_idx": args.mass_token_idx,
        "calorie_token_idx": args.calorie_token_idx,
        "fat_token_idx": args.fat_token_idx,
        "carbohydrate_token_idx": args.carbohydrate_token_idx,
        "protein_token_idx": args.protein_token_idx,
        "sam_ckpt_dir": args.sam_ckpt_dir,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": True,
    }
    args_dict = vars(args)
    for i in range(1, args.max_seg_num + 1):
        added_tokens = tokenizer.add_tokens("[SEG{}]".format(i))
        args_dict['seg_token_idx_%s' % i] = tokenizer("[SEG{}]".format(i), add_special_tokens=False).input_ids[0]

        added_tokens = tokenizer.add_tokens("[MASS{}]".format(i))
        args_dict['mass_token_idx_%s' % i] = tokenizer("[MASS{}]".format(i), add_special_tokens=False).input_ids[0]
        added_tokens = tokenizer.add_tokens("[CAL{}]".format(i))
        args_dict['calorie_token_idx_%s' % i] = tokenizer("[CAL{}]".format(i), add_special_tokens=False).input_ids[0]
        added_tokens = tokenizer.add_tokens("[FAT{}]".format(i))
        args_dict['fat_token_idx_%s' % i] = tokenizer("[FAT{}]".format(i), add_special_tokens=False).input_ids[0]
        added_tokens = tokenizer.add_tokens("[CARB{}]".format(i))
        args_dict['carbohydrate_token_idx_%s' % i] = \
        tokenizer("[CARB{}]".format(i), add_special_tokens=False).input_ids[0]
        added_tokens = tokenizer.add_tokens("[PRO{}]".format(i))
        args_dict['protein_token_idx_%s' % i] = tokenizer("[PRO{}]".format(i), add_special_tokens=False).input_ids[0]

        model_args.update({
            'seg_token_idx_%s' % i: args_dict['seg_token_idx_%s' % i],
            'mass_token_idx_%s' % i: args_dict['mass_token_idx_%s' % i],
            'calorie_token_idx_%s' % i: args_dict['calorie_token_idx_%s' % i],
            'fat_token_idx_%s' % i: args_dict['fat_token_idx_%s' % i],
            'carbohydrate_token_idx_%s' % i: args_dict['carbohydrate_token_idx_%s' % i],
            'protein_token_idx_%s' % i: args_dict['protein_token_idx_%s' % i],
        })

    args = argparse.Namespace(**args_dict)
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )

    # model_args = {
    #     "train_mask_decoder": True,
    #     "out_dim": args.out_dim,
    #     "vision_tower": args.vision_tower,
    #     "seg_token_idx": args.seg_token,
    #     "mass_token": args.mass_token,
    #     "calorie_token": args.calorie_token,
    #     "fat_token": args.fat_token,
    #     "carb_token": args.carb_token,
    #     "protein_token": args.protein_token,
    # }

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = LISAForCausalLM.from_pretrained(
        args.init_ckpt_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    state_dict = torch.load(args.weight, map_location="cpu")
    # model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(state_dict, strict=False)

    model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main(sys.argv[1:])
