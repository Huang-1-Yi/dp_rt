from typing import Tuple
import math
import cv2
import numpy as np

def draw_reticle(img, u, v, label_color):
    """
    Draws a reticle (cross-hair) on the image at the given position on top of
    the original image.
    @param img (In/Out) uint8 3 channel image
    @param u X coordinate (width)
    @param v Y coordinate (height)
    @param label_color tuple of 3 ints for RGB color used for drawing.
    """
    # Cast to int.
    u = int(u)
    v = int(v)

    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform

def get_mask_transform(
        input_res: Tuple[int, int] = (1280, 720),
        output_res: Tuple[int, int] = (640, 480)):
    """专为Mask设计的转换器，处理单通道图像"""
    iw, ih = input_res  # 宽高
    ow, oh = output_res

    # 计算缩放尺寸（保持宽高比）
    aspect_ratio_input = iw / ih
    aspect_ratio_output = ow / oh

    if aspect_ratio_input >= aspect_ratio_output:
        # 输入更宽，按高度缩放
        rh = oh
        rw = int(rh * aspect_ratio_input)
    else:
        # 输入更高，按宽度缩放
        rw = ow
        rh = int(rw / aspect_ratio_input)

    # 计算裁剪区域
    w_slice_start = max(0, (rw - ow) // 2)
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = max(0, (rh - oh) // 2)
    h_slice = slice(h_slice_start, h_slice_start + oh)

    def transform(img: np.ndarray):
        # 验证输入mask的形状 
        assert img.ndim == 2, f"Mask应为单通道 (H,W)，实际维度为{img.ndim}"
        h, w = img.shape
        assert (h, w) == (ih, iw), f"输入尺寸应为高{ih}宽{iw}，实际为高{h}宽{w}"

        # 缩放（使用最近邻插值保持mask的清晰度）
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_NEAREST)
        # 裁剪中心区域
        cropped = resized[h_slice, w_slice]
        return cropped

    return transform

def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution
    out_wh_ratio = out_w / out_h
    
    n_rows = np.arange(n_cameras,dtype=np.int64) + 1
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)
    best_idx = np.argmin(ratio_diff)
    best_n_row = n_rows[best_idx]
    best_n_col = n_cols[best_idx]
    best_cat_wh_ratio = cat_wh_ratio[best_idx]

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # cat is wider
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)
    
    # crop_resolution = (rw, rh)
    return rw, rh, best_n_col, best_n_row