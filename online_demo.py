import argparse
import datetime
import os
import re
import sys
import hashlib
import time
from types import SimpleNamespace

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.llava.constants import LOGDIR
from model.llava.utils import violates_moderation
from utils import conversation as conversation_lib
from model.LISA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.config import Config
from utils.conversation import default_conversation
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, convert_gray_to_color_mask, overlay_rgba_on_rgb,
                         replace_token_with_nutrition_values)


headers = {"User-Agent": "LLaVA Client"}
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


def parse_args(args):
    parser = argparse.ArgumentParser(description="FoodLISA chat")
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--version", default="", required=True)
    parser.add_argument("--host", default='localhost', type=str)
    parser.add_argument("--port", default=7618, type=int, required=False)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=3000, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--share", action="store_true", default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format.",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def add_text(state, text, image, image_process_mode, request: gr.Request):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 2

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            # text = '<Image><image></Image>' + text
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
        state = conversation_lib.conv_templates[args.conv_type].copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return state, state.to_gradio_chatbot(), "", None


def regenerate(state, image_process_mode, request: gr.Request):
    # if type(state.messages[-1][-1]) is tuple and state.messages[-1][-1][0] == '':
    #     state.messages[-1][-2] = None
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]

    # print('prev_msg', prev_human_msg)
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return state, state.to_gradio_chatbot(), "", None


def clear_history(request: gr.Request):
    state = conversation_lib.conv_templates[args.conv_type].copy()
    return state, state.to_gradio_chatbot(), "", None


def inference(state, temperature, top_p, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield state, state.to_gradio_chatbot()
        return

    if len(state.messages) == state.offset + 2:
        new_state = conversation_lib.conv_templates[args.conv_type].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state
    # Construct prompt
    prompt = state.get_prompt()
    prompt = prompt.replace('USER:ASSISTANT: </s>', '')
    # print('prompt', prompt)
    # Define the pattern to match [SEG] followed by one or more digits
    pattern = r"\[SEG\d+\]"
    # Find all matches
    matches = re.findall(pattern, prompt)
    # Count the matches
    previous_masks = len(matches)

    image_np = np.array(state.get_images(return_pil=True)[0])
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
            .unsqueeze(0)
            .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    output_ids, pred_masks, nurtition_output_dict = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("  ", " ").replace(" .", ".").replace(" ,", ",").replace(" g ", "g ").replace(
        "( ", "(")
    text_output = replace_token_with_nutrition_values(text_output, nurtition_output_dict, args.max_seg_num)
    text_output = text_output.split("ASSISTANT: ")[-1].replace('</s>', '')

    save_img = None
    if pred_masks[0].shape[0] > previous_masks:
        colored_mask = torch.zeros(size=image_np.shape)
        # print('num_masks:', pred_masks[0].shape[0])
        for idx in range(pred_masks[0].shape[0]):
            pred_mask = pred_masks[0][idx, :, :]
            pred_mask = pred_mask.detach().cpu().numpy()
            pred_mask = pred_mask > 0
            colored_mask += convert_gray_to_color_mask(pred_mask, color_idx=idx)
        save_img = overlay_rgba_on_rgb(torch.tensor(image_np), torch.tensor(colored_mask) * 255)
        save_img = Image.fromarray(save_img)

    state.messages[-1][-1] = text_output
    if save_img is not None:
        image_process_mode = gr.Radio(
            ["Crop", "Resize", "Pad", "Default"],
            value="Default",
            label="Preprocess for non-square image", visible=False)
        text = ('', save_img, image_process_mode)
        state.append_message(state.roles[0], None)
        state.append_message(state.roles[1], text)
        yield state, state.to_gradio_chatbot()
    else:
        yield state, state.to_gradio_chatbot()


def prepare_demo():
    title_markdown = ("""
    # 🍕 FoodLMM: A Versatile Food Assistant using Large Multi-modal Model
    [[Project Page](coming soon...)] [[Code](coming soon...)] [[Model](coming soon...)]
    """)
    tos_markdown = ("""
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """)
    block_css = """
    #buttons button {
        min-width: min(120px,100%);
    }
    """
    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=4):
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image", visible=False)

                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/examples/00004873.jpg", "Can you identify a dessert item in this image that I could indulge in to satisfy my sweet tooth? Please output segmentation mask."],
                    [f"{cur_dir}/examples/00004401.jpg", "What ingredient in this image might contribute to a unique savoury flavor? Please output segmentation mask."],
                    [f"{cur_dir}/examples/dish_1563909580.jpg", "Can you provide the total nutritional values for the whole dish?"],
                    [f"{cur_dir}/examples/dish_1565034681.jpg", "Could you explain the nutritional breakdown of the food items in the image?"],
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=True) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature", )
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens", )
            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Chatbot",
                    height=950,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="🍕 Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=True)

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False
        ).then(
            inference,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot],
        )
        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox]
        ).then(
            inference,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot],
        )
        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox]
        ).then(
            inference,
            [state, temperature, top_p, max_output_tokens],
            [state, chatbot],
        )
        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, imagebox],
            queue=False
        )

        demo.queue(
            api_open=False
        ).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    configs = Config(args)
    # configs.pretty_print_system()
    args = configs.args
    args = argparse.Namespace(**args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.mass_token_idx = tokenizer("[MASS_TOTAL]", add_special_tokens=False).input_ids[0]
    args.calorie_token_idx = tokenizer("[CAL_TOTAL]", add_special_tokens=False).input_ids[0]
    args.fat_token_idx = tokenizer("[FAT_TOTAL]", add_special_tokens=False).input_ids[0]
    args.carbohydrate_token_idx = tokenizer("[CARB_TOTAL]", add_special_tokens=False).input_ids[0]
    args.protein_token_idx = tokenizer("[PRO_TOTAL]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,
        "seg_token_idx": args.seg_token_idx,
        "mass_token_idx": args.mass_token_idx,
        "calorie_token_idx": args.calorie_token_idx,
        "fat_token_idx": args.fat_token_idx,
        "carbohydrate_token_idx": args.carbohydrate_token_idx,
        "protein_token_idx": args.protein_token_idx,
    }

    args_dict = vars(args)
    for i in range(1, args.max_seg_num + 1):
        args_dict['seg_token_idx_%s' % i] = tokenizer("[SEG{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['mass_token_idx_%s' % i] = tokenizer("[MASS{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['calorie_token_idx_%s' % i] = tokenizer("[CAL{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['fat_token_idx_%s' % i] = tokenizer("[FAT{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['carbohydrate_token_idx_%s' % i] = \
        tokenizer("[CARB{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['protein_token_idx_%s' % i] = tokenizer("[PRO{}]".format(i), add_special_tokens=False).input_ids[0]

        kwargs.update({
            'seg_token_idx_%s' % i: args_dict['seg_token_idx_%s' % i],
            'mass_token_idx_%s' % i: args_dict['mass_token_idx_%s' % i],
            'calorie_token_idx_%s' % i: args_dict['calorie_token_idx_%s' % i],
            'fat_token_idx_%s' % i: args_dict['fat_token_idx_%s' % i],
            'carbohydrate_token_idx_%s' % i: args_dict['carbohydrate_token_idx_%s' % i],
            'protein_token_idx_%s' % i: args_dict['protein_token_idx_%s' % i],
        })
    args = argparse.Namespace(**args_dict)

    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
            args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    prepare_demo()
