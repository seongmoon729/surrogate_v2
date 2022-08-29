import math
import tempfile

from subprocess import Popen, PIPE
from pathlib import Path
from PIL import Image

import ray
import numpy as np


DS_LEVELS = [0, 1, 2, 3]
CODEC_LIST = ['jpeg', 'webp', 'vtm', 'vvc']
JPEG_QUALITIES = [31, 11, 5, 2]
WEBP_QUALITIES = [1, 18, 53, 83, 95]
VTM_QUALITIES  = [47, 42, 37, 32, 27, 22]
VVC_QUALITIES  = [54, 49, 44, 38, 33, 28]


@ray.remote
def ray_codec_fn(x, codec, quality, downscale=0):
    return codec_fn(x, codec, quality, downscale)
    

def codec_fn(x, codec, quality, downscale=0):
    """ Encode & decode input with codec. 
        Args:
            x: 'np.ndarray' with range of [0, 1] and order of (C, H, W).
            codec: 'jpeg', 'webp', or 'vtm'
            quality: a parameter to control quantization level
            downscale: a parameter to control downscaling level
    """
    assert codec in CODEC_LIST

    x = x.copy()
    x *= 255.                      # Denormalize
    x = x.transpose((1, 2, 0))     # (C, H, W) -> (H, W ,C)
    x = x.astype('uint8')
    pil_img = Image.fromarray(x)
    
    pil_img_recon, bpp = run_codec(pil_img, codec, quality, downscale)
    pil_img.close()

    x = np.array(pil_img_recon)
    pil_img_recon.close()

    x = x.astype('float32')
    x = x.transpose((2, 0, 1))     # (H, W, C) -> (C, H, W)
    x /= 255.                      # Normalize to [0, 1]
    return x, bpp


def run_codec(input, codec, q, ds=0, tool_path='/surrogate_v2/tools'):
    assert codec in CODEC_LIST
    assert ds in DS_LEVELS, f"Choose one of {DS_LEVELS}."

    # Set path of binary files & check.
    tool_path = Path(tool_path)
    bin_path = tool_path / 'bin'
    ffmpeg_path = bin_path / 'ffmpeg'
    vtm_path = bin_path / 'EncoderApp_12_0'
    config_path = tool_path / 'configs'
    vtm_config_path = config_path / 'VTM' / 'encoder_intra_vtm_12_0.cfg'
    assert ffmpeg_path.exists()
    assert vtm_path.exists()
    assert vtm_config_path.exists()

    # Define base commands.
    ffmpeg_base_cmd = f"{ffmpeg_path} -y -loglevel error"
    vtm_base_cmd = f"{vtm_path} -c {vtm_config_path}"
    vvenc_cmd = "vvencapp"
    vvdec_cmd = "vvdecapp"
    vvenc_preset = "medium"

    # Make temp directory for processing.
    dst_dir_obj = tempfile.TemporaryDirectory()
    dst_dir = Path(dst_dir_obj.name)

    # Define intermediate file names.
    file_name = 'img'
    yuv_path = dst_dir / (file_name + '.yuv')
    recon_yuv_path = dst_dir / (file_name + '_recon.yuv')
    comp_bin_path = dst_dir / (file_name + '_comp.bin')
    log_path = dst_dir / (file_name + '.log')
    recon_png_path = dst_dir / (file_name + '_recon.png')

    if isinstance(input, str):
        src_img_path = input
        pil_img = Image.open(src_img_path)
    else:
        # Save source input image in temp directory for cmd processing.
        pil_img = input
        src_img_path = dst_dir / 'raw.png'
        pil_img.save(src_img_path)

    # Compute down-scaled or padded size.
    w, h = pil_img.size
    if ds == 0:
        dw, dh = map(lambda x: math.ceil(x / 2) * 2, [w, h])
    else:
        dw, dh = map(lambda x: math.ceil(x * (4 - ds) / 8) * 2, [w, h])

    # Define (down/up scaling) + (RGB <-> YUV conversion) commands.
    if ds == 0:
        down_img2yuv_cmd = (f"{ffmpeg_base_cmd} -i {src_img_path} -vf 'pad={dw}:{dh}'"
                            f" -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_path}")
        yuv2img_up_cmd   = (f"{ffmpeg_base_cmd} -f rawvideo -pix_fmt yuv420p10le -s {dw}x{dh} -src_range 1"
                            f" -i {recon_yuv_path} -frames 1 -pix_fmt rgb24"
                            f" -vf 'crop={w}:{h}:0:0' {recon_png_path}")
    else:
        down_img2yuv_cmd = (f"{ffmpeg_base_cmd} -i {src_img_path} -vf 'scale={dw}:{dh}'"
                            f" -f rawvideo -pix_fmt yuv420p -dst_range 1 {yuv_path}")
        yuv2img_up_cmd   = (f"{ffmpeg_base_cmd} -f rawvideo -pix_fmt yuv420p10le -s {dw}x{dh} -src_range 1"
                            f" -i {recon_yuv_path} -frames 1 -pix_fmt rgb24"
                            f" -vf 'scale={w}:{h}' {recon_png_path}")

    # Define JPEG command.
    jpeg_cmd = (f"{ffmpeg_base_cmd} -f rawvideo -s {dw}x{dh} -pix_fmt yuv420p -i {yuv_path}"
                f" -q:v {q} -f mjpeg {comp_bin_path}" " && " 
                f"{ffmpeg_base_cmd} -i {comp_bin_path} -pix_fmt yuv420p10le {recon_yuv_path}")

    # Define WebP command.
    webp_cmd = (f"{ffmpeg_base_cmd} -f rawvideo -s {dw}x{dh} -pix_fmt yuv420p -i {yuv_path}"
                f" -q:v {q} -f webp {comp_bin_path}" " && "
                f"{ffmpeg_base_cmd} -i {comp_bin_path} -pix_fmt yuv420p10le {recon_yuv_path}")

    # Define VTM command.
    vtm_cmd  = (f"{vtm_base_cmd} -i {yuv_path} -o {recon_yuv_path} -b {comp_bin_path}"
                f" -q {q} --ConformanceWindowMode=1 -wdt {dw} -hgt {dh} -f 1 -fr 1"
                f" --InternalBitDepth=10 > {log_path}")

    # Define VVC command.
    vvc_cmd = (f"{vvenc_cmd} -i {yuv_path} -o {comp_bin_path} -q {q} -s {dw}x{dh} -r 1"
               f" --internal-bitdepth 10 --preset={vvenc_preset} > {log_path} && "
               f"{vvdec_cmd} -b {comp_bin_path} -o {recon_yuv_path} -t 1 > {log_path}")

    # Choose codec.
    if codec == 'jpeg':
        assert q in JPEG_QUALITIES, f"Choose one of {JPEG_QUALITIES}, lower is better."
        codec_cmd = jpeg_cmd
    elif codec == 'webp':
        assert q in WEBP_QUALITIES, f"Choose one of {WEBP_QUALITIES}, higher is better."
        codec_cmd = webp_cmd
    elif codec == 'vtm':
        assert q in VTM_QUALITIES, f"Choose one of {VTM_QUALITIES}, lower is better."
        codec_cmd = vtm_cmd
    elif codec == 'vvc':
        assert q in VVC_QUALITIES, f"Choose one of {VVC_QUALITIES}, lower is better."
        codec_cmd = vvc_cmd

    # Start processing.
    # import time
    # t1 = time.time()
    _run_cmd(down_img2yuv_cmd)  # 1. Down-scale & RGB2YUV.
    # t2 = time.time(); print('down', t2 - t1)
    _run_cmd(codec_cmd)         # 2. Run codec.
    # t3 = time.time(); print('codec', t3 - t2)
    _run_cmd(yuv2img_up_cmd)    # 3. YUV2RGB & Up-scale.
    # t4 = time.time(); print('up', t4 - t3)

    # Generate processing results.
    recon_img = Image.open(recon_png_path)
    bytes = comp_bin_path.stat().st_size
    bpp = bytes * 8 / (h * w)
    return recon_img, bpp


def _run_cmd(cmd):
    proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    _, err = proc.communicate()
    if err:
        print(err)
