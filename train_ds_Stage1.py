import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import torch
import tqdm
import transformers
import wandb
import warnings
from deepspeed.utils.logging import logger
from peft import LoraConfig, get_peft_model

from evaluate.ref_seg import get_test_data_seg, get_data_seg, validate_ref_seg_iou
from model.LISA import LISAForCausalLM
from utils import conversation as conversation_lib
from utils.config import Config
from utils.dataset import HybridDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, is_main_process, is_dist_avail_and_initialized)

logger.setLevel('ERROR')
os.environ['WANDB_API_KEY'] = 'YOUR_KEY'


def parse_args(args):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--local_rank", default=-1, help="path to configuration file.")
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format.",
    )
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    configs = Config(args)
    args = configs.args
    args = argparse.Namespace(**args)
    args.world_size = torch.cuda.device_count()
    args.distributed = args.world_size > 1
    if is_main_process():
        os.makedirs(args.vis_save_path, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.exp_name,
            config=args
        )

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
        args_dict['carbohydrate_token_idx_%s' % i] = tokenizer("[CARB{}]".format(i), add_special_tokens=False).input_ids[0]
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

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    lora_r = args.lora_r
    if lora_r > 0:
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        # print('lora_target_modules', lora_target_modules)
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

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any([x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs",
                                 "mass_head", "calorie_head", "fat_head", "carb_head", "protein_head",
                                 "total_mass_head", "total_calorie_head", "total_fat_head", "total_carb_head", "total_protein_head"]]):
            # print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    # [yyh] data
    processor = transformers.CLIPImageProcessor.from_pretrained(args.vision_tower)
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=processor,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * args.world_size,
        image_size=args.image_size,
        seg_num=args.max_seg_num,
        dataset=args.dataset,
        sample_rate=args.sample_rates,
        num_OOD_per_sample=args.num_OOD_per_sample,
        sem_seg_data=args.sem_seg,
        vqa_data=args.vqa,
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": (args.epochs + 1) * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=True,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    val_data_list_new = []
    if args.val:
        assert args.val_batch_size == 1
        val_data_list = get_data_seg(args, tokenizer, clip_image_processor=processor)
        for test_data in val_data_list:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data['data'], shuffle=False, drop_last=False
            )
            val_loader = torch.utils.data.DataLoader(
                test_data['data'],
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn,
                    tokenizer=tokenizer,
                    conv_type=args.conv_type,
                    use_mm_start_end=True,
                    local_rank=args.local_rank,
                ),
            )
            test_data['data'] = val_loader
            val_data_list_new.append(test_data)

    global_step = 0
    resume_dir = os.path.join(args.ckpt_dir, "ckpt_model", "resume", args.resume)
    if args.auto_resume:
        if os.path.exists(resume_dir):
            args.resume = True
        else:
            args.resume = False
    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(resume_dir)
        resume_info = torch.load(os.path.join(resume_dir, 'resume_info.pth'))
        args.start_epoch = resume_info['epoch']
        global_step = resume_info['global_step']
        if is_main_process():
            print("resume training from {}, start from epoch {}".format(resume_dir, args.start_epoch))

    if args.val:
        for dataset in val_data_list_new:
            torch.cuda.empty_cache()
            if is_main_process():
                print('[{}] evaluating iou of step {}'.format(dataset['name'], global_step))
            validate_ref_seg_iou(
                val_loader=dataset['data'],
                model_engine=model_engine,
                step=global_step,
                args=args, name=dataset['name']
            )

    print('***** Trainig start [is_dist={}] *****'.format(is_dist_avail_and_initialized()))
    for epoch in range(args.start_epoch, args.epochs+1):
        torch.cuda.empty_cache()
        # train for one epoch
        global_step = train(train_loader, model_engine, epoch, global_step, scheduler, args)
        torch.cuda.empty_cache()
        if (epoch % args.val_freq == 0 and epoch > 0) and args.val:
            for dataset in val_data_list_new:
                if is_main_process():
                    print('[{}] evaluating iou of step {}'.format(dataset['name'], global_step))
                validate_ref_seg_iou(
                    val_loader=dataset['data'],
                    model_engine=model_engine,
                    step=global_step,
                    args=args, name=dataset['name']
                )
        if (epoch % args.save_freq == 0 and epoch > 0) or epoch == args.epochs:
            save_dir = os.path.join(args.ckpt_dir, "ckpt_model", args.exp_name, 'train', str(global_step))
            torch.distributed.barrier()
            if is_main_process():
                print('saving ckpt...')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_engine.save_checkpoint(save_dir, tag='global_step{}'.format(global_step))
        if args.auto_resume and (epoch % args.resume_freq == 0 and epoch > 0):
            if is_main_process():
                print('saving resume_ckpt ...')
            torch.distributed.barrier()
            if os.path.exists(resume_dir) and is_main_process():
                shutil.rmtree(resume_dir)
            torch.distributed.barrier()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_engine.save_checkpoint(resume_dir, tag='global_step{}'.format(global_step))
                if is_main_process():
                    resume_info = {
                        'epoch': epoch + 1,
                        'global_step': global_step + 1
                    }
                    torch.save(resume_info, f=os.path.join(resume_dir, 'resume_info.pth'))

    if is_main_process():
        # wandb.finish()
        generate_ckpt_hf(args.init_ckpt_dir, os.path.join(args.ckpt_dir, "ckpt_model", args.exp_name), args.cfg_file)


def train(train_loader, model, epoch, global_step, scheduler, args):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Overall_Loss", ":.4f")
    ce_losses = AverageMeter("Overall_CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("Overall_MaskLoss", ":.4f")

    nutri_losses = AverageMeter("Overall_nutri_Loss", ":.4f")

    nutri_mae_losses = AverageMeter("Nutri_Loss", ":.4f")
    mass_mae_losses = AverageMeter("Nutri_mass_loss", ":.4f")
    calorie_mae_losses = AverageMeter("Nutri_calorie_loss", ":.4f")
    fat_mae_losses = AverageMeter("Nutri_fat_loss", ":.4f")
    carb_mae_losses = AverageMeter("Nutri_carb_loss", ":.4f")
    protein_mae_losses = AverageMeter("Nutri_protein_loss", ":.4f")

    total_nutri_mae_losses = AverageMeter("Total_nutri_Loss", ":.4f")
    total_mass_mae_losses = AverageMeter("Total_nutri_mass_loss", ":.4f")
    total_calorie_mae_losses = AverageMeter("Total_nutri_calorie_loss", ":.4f")
    total_fat_mae_losses = AverageMeter("Total_nutri_fat_loss", ":.4f")
    total_carb_mae_losses = AverageMeter("Total_nutri_carb_loss", ":.4f")
    total_protein_mae_losses = AverageMeter("Total_nutri_protein_loss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses
        ],
        prefix="[Epoch={}, global_step={}]".format(epoch, global_step),
    )
    # switch to train mode
    model.train()
    train_iter = iter(train_loader)
    end = time.time()
    for step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)
            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            # try:
            output_dict = model(**input_dict)
            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]

            nutri_loss = output_dict["nutri_loss"]
            nutri_mae_loss = output_dict["nutri_mae_loss"]
            mass_loss = output_dict["mass_loss"]
            calorie_loss = output_dict["calorie_loss"]
            fat_loss = output_dict["fat_loss"]
            carb_loss = output_dict["carb_loss"]
            protein_loss = output_dict["protein_loss"]
            total_nutri_mae_loss = output_dict["total_nutri_mae_loss"]
            total_mass_loss = output_dict["total_mass_loss"]
            total_calorie_loss = output_dict["total_calorie_loss"]
            total_fat_loss = output_dict["total_fat_loss"]
            total_carb_loss = output_dict["total_carb_loss"]
            total_protein_loss = output_dict["total_protein_loss"]

            model.backward(loss)
            model.step()

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))

            nutri_losses.update(nutri_loss.item(), input_dict["images"].size(0))
            nutri_mae_losses.update(nutri_mae_loss.item(), input_dict["images"].size(0))
            mass_mae_losses.update(mass_loss.item(), input_dict["images"].size(0))
            calorie_mae_losses.update(calorie_loss.item(), input_dict["images"].size(0))
            fat_mae_losses.update(fat_loss.item(), input_dict["images"].size(0))
            carb_mae_losses.update(carb_loss.item(), input_dict["images"].size(0))
            protein_mae_losses.update(protein_loss.item(), input_dict["images"].size(0))
            total_nutri_mae_losses.update(total_nutri_mae_loss.item(), input_dict["images"].size(0))
            total_mass_mae_losses.update(total_mass_loss.item(), input_dict["images"].size(0))
            total_calorie_mae_losses.update(total_calorie_loss.item(), input_dict["images"].size(0))
            total_fat_mae_losses.update(total_fat_loss.item(), input_dict["images"].size(0))
            total_carb_mae_losses.update(total_carb_loss.item(), input_dict["images"].size(0))
            total_protein_mae_losses.update(total_protein_loss.item(), input_dict["images"].size(0))

            del input_dict, output_dict
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        global_step += 1
        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

                nutri_losses.all_reduce()
                nutri_mae_losses.all_reduce()
                mass_mae_losses.all_reduce()
                calorie_mae_losses.all_reduce()
                fat_mae_losses.all_reduce()
                carb_mae_losses.all_reduce()
                protein_mae_losses.all_reduce()
                total_nutri_mae_losses.all_reduce()
                total_mass_mae_losses.all_reduce()
                total_calorie_mae_losses.all_reduce()
                total_fat_mae_losses.all_reduce()
                total_carb_mae_losses.all_reduce()
                total_protein_mae_losses.all_reduce()

            curr_lr = scheduler.get_last_lr()
            progress.prefix = "[Epoch={}, global_step={}]".format(epoch, global_step)
            progress.display(step + 1)
            log_dict = {
                "overall_losses/Overall_Loss": losses.avg,
                "overall_losses/CeLoss": ce_losses.avg,
                "mask_losses/mask_bce_loss": mask_bce_losses.avg,
                "mask_losses/mask_dice_loss": mask_dice_losses.avg,
                "overall_losses/MaskLoss": mask_losses.avg,
                "metrics/total_secs_per_batch": batch_time.avg,
                "metrics/data_secs_per_batch": data_time.avg,

                "overall_losses/Nutri_Loss": nutri_losses.avg,

                "ing_nutrition_losses/all_nutri_losses": nutri_mae_losses.avg,
                "ing_nutrition_losses/mass_mae_losses": mass_mae_losses.avg,
                "ing_nutrition_losses/calorie_mae_losses": calorie_mae_losses.avg,
                "ing_nutrition_losses/fat_mae_losses": fat_mae_losses.avg,
                "ing_nutrition_losses/carb_mae_losses": carb_mae_losses.avg,
                "ing_nutrition_losses/protein_mae_losses": protein_mae_losses.avg,

                "total_nutrition_losses/total_all_nutri_losses": total_nutri_mae_losses.avg,
                "total_nutrition_losses/total_mass_mae_losses": total_mass_mae_losses.avg,
                "total_nutrition_losses/total_calorie_mae_losses": total_calorie_mae_losses.avg,
                "total_nutrition_losses/total_fat_mae_losses": total_fat_mae_losses.avg,
                "total_nutrition_losses/total_carb_mae_losses": total_carb_mae_losses.avg,
                "total_nutrition_losses/total_protein_mae_losses": total_protein_mae_losses.avg,

                "train/lr": curr_lr[0],
            }
            if is_main_process():
                wandb.log(log_dict, step=global_step)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

            nutri_losses.reset()
            nutri_mae_losses.reset()
            mass_mae_losses.reset()
            calorie_mae_losses.reset()
            fat_mae_losses.reset()
            carb_mae_losses.reset()
            protein_mae_losses.reset()
            total_nutri_mae_losses.reset()
            total_mass_mae_losses.reset()
            total_calorie_mae_losses.reset()
            total_fat_mae_losses.reset()
            total_carb_mae_losses.reset()
            total_protein_mae_losses.reset()

    return global_step


def find_linear_layers(model, lora_target_modules):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, cls
                       ) and all([x not in name
                                  for x in [
                                      "visual_model",
                                      "vision_tower",
                                      "mm_projector",
                                      "text_hidden_fcs",
                                  ]
                                  ]) and any([x in name for x in lora_target_modules])):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


def generate_ckpt_hf(init_ckpt_dir, ckpt_dir, cfg_file):
    ckpt_dir_train = os.path.join(ckpt_dir, 'train')
    global_steps = os.listdir(ckpt_dir_train)
    global_steps = sorted(global_steps, key=int)
    for global_step in global_steps:
        path = os.path.join(ckpt_dir_train, global_step)
        save_path = os.path.join(ckpt_dir, global_step)

        tag = 'global_step{}'.format(global_step)

        if not os.path.exists('{}/pytorch_model.bin'.format(path)):
            print('generating ckpt_hf of {}->{}'.format(path, save_path))
            cmd = ['python {}/zero_to_fp32.py {} {}/pytorch_model.bin -t {}'.format(path, path, path, tag)]
            os.system(' '.join(cmd))

        print('merge {} ckpt to '.format('{}/pytorch_model.bin'.format(path)), save_path)
        os.makedirs(save_path, exist_ok=True)
        cmd = [
            'CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py --weight="{}/pytorch_model.bin" --save_path="{}"'.format(
                cfg_file, path, save_path)]
        os.system(' '.join(cmd))


if __name__ == "__main__":
    main(sys.argv[1:])
