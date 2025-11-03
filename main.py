# For Editing with SD3
import os
import time
import datetime
import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusion3InstructPix2PixPipeline as SD3Pipeline
from diffusers.utils import load_image

# Input
ori_img_dir  = "../WP1/output_20251009_try_default_general_image_jpeg_scale/img_save_dir/img_blended"
mask_img_dir = "../MAT/test_sets/img_blended_masks"

# Output
gen_img_dir  = "./img_gen_output_20251009_try_default_general_image_jpeg_scale"
inp_img_dir  = "./img_inpaint_output_20251009_try_default_general_image_jpeg_scale"
mer_img_dir  = "./img_merge_output_20251009_try_default_general_image_jpeg_scale"

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def load_model_pipeline():
    model_name_or_path = "BleachNick/SD3_UltraEdit_w_mask"
    pipe = SD3Pipeline.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def load_ori_img(ori_img_name):
    ori_img = load_image(os.path.join(ori_img_dir, ori_img_name))
    w, h = ori_img.size
    if w % 16 != 0 or h % 16 != 0:
        # If width or height is not multiples of 16, it will report error for image size 270*360:
        #   The size of tensor a (33) must match the size of tensor b (32) at non-singleton dimension 2
        # Therefore, we resize the width and height.
        w -= w % 16
        h -= h % 16
        ori_img = ori_img.resize((w, h))
    return ori_img

def load_mask_img(mask_file_name, ori_img):
    #  mask_img = load_image(os.path.join(ori_img_dir, "mask_img.png")).resize(ori_img.size)

    mask_file_path = os.path.join(mask_img_dir, mask_file_name)

    # Since original image may be resized, we resize the mask
    mask_img = Image.open(mask_file_path).resize(ori_img.size)

    # In mask image, it needs black background and white foreground.
    # However, "../MAT/test_sets/img_blended_masks/", it has white background and black foreground.
    # Therefore, we invert the color
    mask_img = ImageOps.invert(mask_img)
    bands = mask_img.getbands()
    num_channels = len(bands)
    if num_channels == 1:
        mask_img = mask_img.convert("RGB")  # from 1 channel to 3 channels
    return mask_img

def save_images(ori_img_name, ori_img, mask_img, generated_image):
    generated_image.save(os.path.join(gen_img_dir, ori_img_name))
    inpaint_img = Image.new("RGB", ori_img.size)
    inpaint_img.paste(ori_img, (0, 0))
    mask_l = mask_img.convert("L")
    inpaint_img.paste(generated_image, (0, 0), mask_l)
    inpaint_img.save(os.path.join(inp_img_dir, ori_img_name))

    # concatenate original-image, mask-image, inpaint-image and generated-image
    w, h = ori_img.size
    merge_img = Image.new("RGB", (w * 2, h * 2))
    merge_img.paste(ori_img, (0, 0))
    merge_img.paste(mask_img, (w, 0))
    merge_img.paste(inpaint_img, (0, h))
    merge_img.paste(generated_image, (w, h))
    merge_img.save(os.path.join(mer_img_dir, ori_img_name))

def main():
    os.makedirs(gen_img_dir, exist_ok=True)
    os.makedirs(inp_img_dir, exist_ok=True)
    os.makedirs(mer_img_dir, exist_ok=True)
    ori_img_name_list = os.listdir(ori_img_dir)
    ori_img_name_list.sort()
    img_cnt = len(ori_img_name_list)
    print(f"ori_img_dir: {ori_img_dir}")
    print(f"  image count   : {img_cnt}")
    print(f"  image list[0] : {ori_img_name_list[0]}")
    print(f"  image list[-1]: {ori_img_name_list[-1]}")
    mask_name_list = os.listdir(mask_img_dir)
    mask_name_list.sort()
    mask_cnt = len(mask_name_list)
    print(f"mask_img_dir: {mask_img_dir}")
    print(f"  mask count    : {mask_cnt}")
    print(f"  mask list[0]  : {mask_name_list[0]}")
    print(f"  mask list[-1] : {mask_name_list[-1]}")
    if img_cnt != mask_cnt:
        raise ValueError(f"img_cnt not match mask_cnt {img_cnt} vs {mask_cnt}")
    model_pipeline = load_model_pipeline()
    start_time = time.time()
    for i, (ori_img_name, mask_file_name) in enumerate(zip(ori_img_name_list, mask_name_list)):
        if i < 10 or i < 100 and i % 10 == 0 or i % 50 == 0 or i == img_cnt - 1:
            dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            elp, eta = get_time_ttl_and_eta(start_time, i, img_cnt)
            print(f"[{dt_str}] {i:04d}/{img_cnt}: {ori_img_name}. elp:{elp}, eta:{eta}")
        ori_img = load_ori_img(ori_img_name)
        mask_img = load_mask_img(mask_file_name, ori_img)
        generated_image = model_pipeline(
            "", # prompt
            image=ori_img,
            mask_img=mask_img,
            negative_prompt="",
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
        ).images[0]
        save_images(ori_img_name, ori_img, mask_img, generated_image)
    # for

if __name__ == "__main__":
    main()
