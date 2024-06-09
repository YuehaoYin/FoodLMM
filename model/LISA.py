from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


def n5kloss(
    gt: List[list],
    output: torch.Tensor,
):
    new_gt_list = []
    for idx in range(len(gt)):
        new_gt_list.extend(gt[idx])
    gt_tensor = torch.Tensor(new_gt_list).reshape(-1)
    pred = output.reshape(-1)
    gt = gt_tensor[:pred.shape[0]]
    gt = gt.to(output.device)

    loss = torch.abs(pred - gt) + 0.001 * (pred - gt) ** 2
    return torch.sum(loss) / (loss.shape[0] + 1e-8)


class LisaMetaModel:
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)
        self.config = config
        self.sam_ckpt_dir = kwargs.get("sam_ckpt_dir", None)
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self._set_sam()
        else:
            self._set_sam()
            self.initialize_lisa_modules(self.config)

    def _set_sam(self):
        self.visual_model = build_sam_vit_h(self.sam_ckpt_dir)
        # self.visual_model = build_sam_vit_h()
        in_dim = self.config.hidden_size
        out_dim = self.config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])

    def initialize_lisa_modules(self, config):
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        # print('--------kwargs: \n', kwargs)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
        else:
            config.mm_vision_tower = config.vision_tower

        self.max_seg_num = kwargs.pop("max_seg_num", 20)

        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 1.)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", .5)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", 2.)
        self.nutrition_loss_weight = kwargs.pop("nutrition_loss_weight", .1)

        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.mass_token_idx = kwargs.pop("mass_token_idx")
        self.calorie_token_idx = kwargs.pop("calorie_token_idx")
        self.fat_token_idx = kwargs.pop("fat_token_idx")
        self.carbohydrate_token_idx = kwargs.pop("carbohydrate_token_idx")
        self.protein_token_idx = kwargs.pop("protein_token_idx")
        for i in range(1, self.max_seg_num+1):
            setattr(self, 'seg_token_idx_%s' % i, kwargs.pop('seg_token_idx_%s' % i))
            setattr(self, 'mass_token_idx_%s' % i, kwargs.pop('mass_token_idx_%s' % i))
            setattr(self, 'calorie_token_idx_%s' % i, kwargs.pop('calorie_token_idx_%s' % i))
            setattr(self, 'fat_token_idx_%s' % i, kwargs.pop('fat_token_idx_%s' % i))
            setattr(self, 'carbohydrate_token_idx_%s' % i, kwargs.pop('carbohydrate_token_idx_%s' % i))
            setattr(self, 'protein_token_idx_%s' % i, kwargs.pop('protein_token_idx_%s' % i))

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mass_head = nn.Linear(256, 1)
        self.calorie_head = nn.Linear(256, 1)
        self.fat_head = nn.Linear(256, 1)
        self.carb_head = nn.Linear(256, 1)
        self.protein_head = nn.Linear(256, 1)
        self.total_mass_head = nn.Linear(256, 1)
        self.total_calorie_head = nn.Linear(256, 1)
        self.total_fat_head = nn.Linear(256, 1)
        self.total_carb_head = nn.Linear(256, 1)
        self.total_protein_head = nn.Linear(256, 1)
        # Initialize weights and apply final processing
        self.post_init()

    def get_token_mask(self, input_ids, device, is_train=True):
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        total_mass_token_mask = input_ids[:, 1:] == self.mass_token_idx
        total_calorie_token_mask = input_ids[:, 1:] == self.calorie_token_idx
        total_fat_token_mask = input_ids[:, 1:] == self.fat_token_idx
        total_carb_token_mask = input_ids[:, 1:] == self.carbohydrate_token_idx
        total_protein_token_mask = input_ids[:, 1:] == self.protein_token_idx

        mass_token_mask = torch.zeros_like(total_mass_token_mask).bool()
        calorie_token_mask = torch.zeros_like(total_mass_token_mask).bool()
        fat_token_mask = torch.zeros_like(total_mass_token_mask).bool()
        carb_token_mask = torch.zeros_like(total_mass_token_mask).bool()
        protein_token_mask = torch.zeros_like(total_mass_token_mask).bool()
        for i in range(1, self.max_seg_num+1):
            seg_token_mask = seg_token_mask | (input_ids[:, 1:] == getattr(self, 'seg_token_idx_%s' % i))
            mass_token_mask = mass_token_mask | (input_ids[:, 1:] == getattr(self, 'mass_token_idx_%s' % i))
            calorie_token_mask = calorie_token_mask | (input_ids[:, 1:] == getattr(self, 'calorie_token_idx_%s' % i))
            fat_token_mask = fat_token_mask | (input_ids[:, 1:] == getattr(self, 'fat_token_idx_%s' % i))
            carb_token_mask = carb_token_mask | (input_ids[:, 1:] == getattr(self, 'carbohydrate_token_idx_%s' % i))
            protein_token_mask = protein_token_mask | (input_ids[:, 1:] == getattr(self, 'protein_token_idx_%s' % i))

        if is_train:
            seg_token_mask = torch.cat([seg_token_mask, torch.zeros((seg_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            mass_token_mask = torch.cat([mass_token_mask, torch.zeros((mass_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            calorie_token_mask = torch.cat([calorie_token_mask, torch.zeros((calorie_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            fat_token_mask = torch.cat([fat_token_mask, torch.zeros((fat_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            carb_token_mask = torch.cat([carb_token_mask, torch.zeros((carb_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            protein_token_mask = torch.cat([protein_token_mask, torch.zeros((protein_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            total_mass_token_mask = torch.cat([total_mass_token_mask, torch.zeros((total_mass_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            total_calorie_token_mask = torch.cat([total_calorie_token_mask, torch.zeros((total_calorie_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            total_fat_token_mask = torch.cat([total_fat_token_mask, torch.zeros((total_fat_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            total_carb_token_mask = torch.cat([total_carb_token_mask, torch.zeros((total_carb_token_mask.shape[0], 1)).bool().to(device)], dim=1)
            total_protein_token_mask = torch.cat([total_protein_token_mask, torch.zeros((total_protein_token_mask.shape[0], 1)).bool().to(device)], dim=1)

        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 255)).bool().to(device), seg_token_mask], dim=1)
        mass_token_mask = torch.cat([torch.zeros((mass_token_mask.shape[0], 255)).bool().to(device), mass_token_mask], dim=1)
        calorie_token_mask = torch.cat([torch.zeros((calorie_token_mask.shape[0], 255)).bool().to(device), calorie_token_mask], dim=1)
        fat_token_mask = torch.cat([torch.zeros((fat_token_mask.shape[0], 255)).bool().to(device), fat_token_mask], dim=1)
        carb_token_mask = torch.cat([torch.zeros((carb_token_mask.shape[0], 255)).bool().to(device), carb_token_mask], dim=1)
        protein_token_mask = torch.cat([torch.zeros((protein_token_mask.shape[0], 255)).bool().to(device), protein_token_mask], dim=1)

        total_mass_token_mask = torch.cat([torch.zeros((total_mass_token_mask.shape[0], 255)).bool().to(device), total_mass_token_mask], dim=1)
        total_calorie_token_mask = torch.cat([torch.zeros((total_calorie_token_mask.shape[0], 255)).bool().to(device), total_calorie_token_mask], dim=1)
        total_fat_token_mask = torch.cat([torch.zeros((total_fat_token_mask.shape[0], 255)).bool().to(device), total_fat_token_mask], dim=1)
        total_carb_token_mask = torch.cat([torch.zeros((total_carb_token_mask.shape[0], 255)).bool().to(device), total_carb_token_mask], dim=1)
        total_protein_token_mask = torch.cat([torch.zeros((total_protein_token_mask.shape[0], 255)).bool().to(device), total_protein_token_mask], dim=1)

        return seg_token_mask, total_mass_token_mask, total_calorie_token_mask, total_fat_token_mask, total_carb_token_mask, total_protein_token_mask, mass_token_mask, calorie_token_mask, fat_token_mask, carb_token_mask, protein_token_mask

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        mass_gt_list: List[list],
        calorie_gt_list: List[list],
        fat_gt_list: List[list],
        carb_gt_list: List[list],
        protein_gt_list: List[list],
        total_mass_gt_list: List[list],
        total_calorie_gt_list: List[list],
        total_fat_gt_list: List[list],
        total_carb_gt_list: List[list],
        total_protein_gt_list: List[list],
        inference: bool = False,
        **kwargs,
    ):
        device = images.device
        image_embeddings_SAM = self.get_visual_embs(images)
        batch_size = image_embeddings_SAM.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask, total_mass_token_mask, total_calorie_token_mask, total_fat_token_mask, total_carb_token_mask, total_protein_token_mask, mass_token_mask, calorie_token_mask, fat_token_mask, carb_token_mask, protein_token_mask = self.get_token_mask(input_ids, device)

        images_clip_list = []
        for i in range(batch_size):
            start_i, end_i = offset[i], offset[i + 1]
            images_clip_i = (
                images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
            )
            images_clip_list.append(images_clip_i)
        images_clip = torch.cat(images_clip_list, dim=0)

        hidden_states = []
        pred_masks = []
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
        )

        output_hidden_states = output.hidden_states
        assert len(self.model.text_hidden_fcs) == 1

        if inference:
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
        else:
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_seg_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        # seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        seg_token_offset = torch.cat([torch.zeros(1).long().to(device), seg_token_offset], dim=0)
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        # i in range( batch_size )
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_seg_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        # for i in range( batch_size )
        # [yyh] every sample
        for i in range(len(pred_embeddings)):
            pred_mask_per_sample = []
            if pred_embeddings[i].shape[0] == 0:
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings_SAM[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])
            else:
                for j in range(pred_embeddings[i].shape[0]):
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i][j, :].unsqueeze(0).unsqueeze(0),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings_SAM[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=label_list[i].shape,
                    )
                    # print('sam pred_mask', pred_mask.shape)
                    pred_mask_per_sample.append(pred_mask[0])
                pred_mask_per_sample = torch.cat(pred_mask_per_sample, dim=0)
                pred_masks.append(pred_mask_per_sample)
        model_output = output
        gt_masks = masks_list

        mass_pred_embeddings = last_hidden_state[mass_token_mask]
        calorie_pred_embeddings = last_hidden_state[calorie_token_mask]
        fat_pred_embeddings = last_hidden_state[fat_token_mask]
        carb_pred_embeddings = last_hidden_state[carb_token_mask]
        protein_pred_embeddings = last_hidden_state[protein_token_mask]
        total_mass_pred_embeddings = last_hidden_state[total_mass_token_mask]
        total_calorie_pred_embeddings = last_hidden_state[total_calorie_token_mask]
        total_fat_pred_embeddings = last_hidden_state[total_fat_token_mask]
        total_carb_pred_embeddings = last_hidden_state[total_carb_token_mask]
        total_protein_pred_embeddings = last_hidden_state[total_protein_token_mask]

        # qhy
        mass_output = self.mass_head(mass_pred_embeddings)
        calorie_output = self.calorie_head(calorie_pred_embeddings)
        fat_output = self.fat_head(fat_pred_embeddings)
        carb_output = self.carb_head(carb_pred_embeddings)
        protein_output = self.protein_head(protein_pred_embeddings)
        mass_loss = n5kloss(mass_gt_list, mass_output)
        calorie_loss = n5kloss(calorie_gt_list, calorie_output)
        fat_loss = n5kloss(fat_gt_list, fat_output)
        carb_loss = n5kloss(carb_gt_list, carb_output)
        protein_loss = n5kloss(protein_gt_list, protein_output)

        total_mass_output = self.total_mass_head(total_mass_pred_embeddings)
        total_calorie_output = self.total_calorie_head(total_calorie_pred_embeddings)
        total_fat_output = self.total_fat_head(total_fat_pred_embeddings)
        total_carb_output = self.total_carb_head(total_carb_pred_embeddings)
        total_protein_output = self.total_protein_head(total_protein_pred_embeddings)
        total_mass_loss = n5kloss(total_mass_gt_list, total_mass_output)
        total_calorie_loss = n5kloss(total_calorie_gt_list, total_calorie_output)
        total_fat_loss = n5kloss(total_fat_gt_list, total_fat_output)
        total_carb_loss = n5kloss(total_carb_gt_list, total_carb_output)
        total_protein_loss = n5kloss(total_protein_gt_list, total_protein_output)

        nurtition_output_dict = {
            'mass_output': mass_output,
            'calorie_output': calorie_output,
            'fat_output': fat_output,
            'carb_output': carb_output,
            'protein_output': protein_output,
            'total_mass_output': total_mass_output,
            'total_calorie_output': total_calorie_output,
            'total_fat_output': total_fat_output,
            'total_carb_output': total_carb_output,
            'total_protein_output': total_protein_output
        }

        if inference:
            return {
                "output_ids": output.logits,
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
                "nurtition_output_dict": nurtition_output_dict,
            }

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        loss = 0
        mask_bce_loss = 0
        mask_dice_loss = 0
        mask_loss = 0
        num_masks = 0
        nutri_loss = 0
        # mass_loss, calorie_loss, fat_loss, carb_loss, protein_loss = 0, 0, 0, 0, 0
        for mask_idx in range(len(pred_masks)):
            gt_mask = gt_masks[mask_idx]
            pred_mask = pred_masks[mask_idx]
            mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        nutri_mae_loss = mass_loss + calorie_loss + fat_loss + carb_loss + protein_loss
        total_nutri_mae_loss = total_mass_loss + total_calorie_loss + total_fat_loss + total_carb_loss + total_protein_loss
        nutri_loss = self.nutrition_loss_weight * (nutri_mae_loss + total_nutri_mae_loss)
        loss = ce_loss + mask_loss + nutri_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "nutri_loss": nutri_loss,
            "nutri_mae_loss": nutri_mae_loss,
            "mass_loss": mass_loss,
            "calorie_loss": calorie_loss,
            "fat_loss": fat_loss,
            "carb_loss": carb_loss,
            "protein_loss": protein_loss,
            "total_nutri_mae_loss": total_nutri_mae_loss,
            "total_mass_loss": total_mass_loss,
            "total_calorie_loss": total_calorie_loss,
            "total_fat_loss": total_fat_loss,
            "total_carb_loss": total_carb_loss,
            "total_protein_loss": total_protein_loss,
        }

    @torch.no_grad()
    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=3255,
        tokenizer=None,
        **kwargs,
    ):
        assert images.shape[0] == 1
        device = images.device
        outputs = self.generate(
            images=images_clip,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        output_hidden_states = outputs.hidden_states[-1]
        output_ids = outputs.sequences

        seg_token_mask, total_mass_token_mask, total_calorie_token_mask, total_fat_token_mask, total_carb_token_mask, total_protein_token_mask, mass_token_mask, calorie_token_mask, fat_token_mask, carb_token_mask, protein_token_mask = self.get_token_mask(output_ids, device, is_train=False)

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # import pdb
        # pdb.set_trace()

        mass_pred_embeddings = last_hidden_state[mass_token_mask]
        calorie_pred_embeddings = last_hidden_state[calorie_token_mask]
        fat_pred_embeddings = last_hidden_state[fat_token_mask]
        carb_pred_embeddings = last_hidden_state[carb_token_mask]
        protein_pred_embeddings = last_hidden_state[protein_token_mask]
        mass_output = self.mass_head(mass_pred_embeddings)
        calorie_output = self.calorie_head(calorie_pred_embeddings)
        fat_output = self.fat_head(fat_pred_embeddings)
        carb_output = self.carb_head(carb_pred_embeddings)
        protein_output = self.protein_head(protein_pred_embeddings)
        total_mass_pred_embeddings = last_hidden_state[total_mass_token_mask]
        total_calorie_pred_embeddings = last_hidden_state[total_calorie_token_mask]
        total_fat_pred_embeddings = last_hidden_state[total_fat_token_mask]
        total_carb_pred_embeddings = last_hidden_state[total_carb_token_mask]
        total_protein_pred_embeddings = last_hidden_state[total_protein_token_mask]
        total_mass_output = self.total_mass_head(total_mass_pred_embeddings)
        total_calorie_output = self.total_calorie_head(total_calorie_pred_embeddings)
        total_fat_output = self.total_fat_head(total_fat_pred_embeddings)
        total_carb_output = self.total_carb_head(total_carb_pred_embeddings)
        total_protein_output = self.total_protein_head(total_protein_pred_embeddings)

        nurtition_output_dict = {
            'mass_output': mass_output,
            'calorie_output': calorie_output,
            'fat_output': fat_output,
            'carb_output': carb_output,
            'protein_output': protein_output,
            'total_mass_output': total_mass_output,
            'total_calorie_output': total_calorie_output,
            'total_fat_output': total_fat_output,
            'total_carb_output': total_carb_output,
            'total_protein_output': total_protein_output
        }

        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        image_embeddings_SAM = self.get_visual_embs(images)

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            pred_mask_per_sample = []
            if pred_embeddings[i].shape[0] == 0:
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings_SAM[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])
            else:
                for j in range(pred_embeddings[i].shape[0]):
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=pred_embeddings[i][j, :].unsqueeze(0).unsqueeze(0),
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                        image_embeddings=image_embeddings_SAM[i].unsqueeze(0),
                        image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=original_size_list[i],
                    )
                    pred_mask_per_sample.append(pred_mask[0])
                pred_mask_per_sample = torch.cat(pred_mask_per_sample, dim=0)
                pred_masks.append(pred_mask_per_sample)

        return output_ids, pred_masks, nurtition_output_dict
