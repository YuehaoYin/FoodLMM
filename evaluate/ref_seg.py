import torch
import tqdm
import wandb

from utils.S2_f103_dataset import S2F103Dataset
from utils.sem_seg_dataset import SemSegDataset
from utils.utils import AverageMeter, Summary, dict_to_cuda, intersectionAndUnionGPU, is_main_process


def get_data_seg(args, tokenizer, clip_image_processor, seg_nums: list = None, reasoning_data=False, model_name=''):
    if seg_nums is None:
        seg_nums = [1]
    vision_tower = clip_image_processor
    test_dataset_list = []
    if reasoning_data:
        test_dataset_list.extend(get_test_data_seg(
            base_image_dir=args.dataset_dir,
            vision_tower=vision_tower,
            image_size=args.image_size,
            seg_num=seg_nums,
            sem_seg_data=args.sem_seg,
            reasoning_data=True,
            model_name=model_name
        ))
    else:
        for seg_num in seg_nums:
            test_dataset_list.extend(get_test_data_seg(
                base_image_dir=args.dataset_dir,
                vision_tower=vision_tower,
                image_size=args.image_size,
                seg_num=seg_num,
                sem_seg_data=args.sem_seg,
                reasoning_data=False
            ))
    return test_dataset_list


def get_test_data_seg(
        base_image_dir,
        vision_tower,
        image_size: int = 224,
        seg_num=1,
        sem_seg_data="foodseg103||uecfoodpix",
        reasoning_data=False,
        model_name=''
):
    datasets = []
    if reasoning_data:
        print('test reasoning')
        if is_main_process():
            # define a metric we are interested in the maximum of
            wandb.define_metric("{}/giou".format('S2_reasoningSeg'), summary="max")
            wandb.define_metric("{}/ciou".format('S2_reasoningSeg'), summary="max")
        dataset_dict = {'name': 'reason_seg',
                        'seg_num': seg_num,
                        'data': S2F103Dataset(
                            base_image_dir=base_image_dir,
                            vision_tower=vision_tower,
                            image_size=image_size,
                            seg_num=seg_num,
                            is_val=True,
                            model_name=model_name
                        ),
                        'type': 'sem_seg'
                        }
        datasets.append(dataset_dict)
    else:
        dataset_names = sem_seg_data.split("||")
        for name in dataset_names:
            if is_main_process():
                # define a metric we are interested in the maximum of
                wandb.define_metric("{}/giou@{}".format(name, seg_num), summary="max")
                wandb.define_metric("{}/ciou@{}".format(name, seg_num), summary="max")
            dataset_dict = {'name': name,
                            'seg_num': seg_num,
                            'data': SemSegDataset(
                                base_image_dir=base_image_dir,
                                vision_tower=vision_tower,
                                image_size=image_size,
                                seg_num=seg_num,
                                num_OOD_per_sample=0,
                                sem_seg_data=name,
                                is_val=True
                            ),
                            'type': 'sem_seg'
                            }
            datasets.append(dataset_dict)

    return datasets


@torch.no_grad()
def validate_ref_seg_iou(val_loader, model_engine, step, args, name='None', seg_num=1):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
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
        output_dict = model_engine(**input_dict)
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"]
        batch_size = len(pred_masks)
        for id_e in range(batch_size):
            masks_list = masks_list[id_e].int()
            output_list = (pred_masks[id_e] > 0).int()
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(union)
            # acc_iou_meter.update(acc_iou, n=masks_list.shape[0])
            acc_iou_meter.update(acc_iou)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if is_main_process():
        wandb.log({"{}/giou@{}".format(name, seg_num): giou, "{}/ciou@{}".format(name, seg_num): ciou}, step=int(step))
        print("[{} of step {}] giou@{}: {:.4f}, ciou@{}: {:.4f}".format(name, step, seg_num, giou, seg_num, ciou))
    del intersection_meter, union_meter, acc_iou_meter, _
