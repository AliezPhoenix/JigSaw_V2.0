import cv2 as cv


def image_preprocess(image, clip_limit=2.0, tile_size=8, denoise_h=3):
    """
    先降噪，再用 CLAHE 调节亮度/局部对比度。

    降噪：灰度图用 fastNlMeansDenoising；BGR 用 fastNlMeansDenoisingColored。
    亮度：在灰度上 CLAHE，BGR 输入时输出仍为三通道 BGR uint8。

    Args:
        image: 输入图像，BGR 三通道或单通道灰度，uint8。
        clip_limit: CLAHE 对比度限制，常用 1.0～4.0。
        tile_size: CLAHE 分块边长（至少为 2）。
        denoise_h: 降噪强度，越大去噪越强、细节可能变软，常用 1～10。

    Returns:
        与输入同形状、同 dtype 的 uint8 图像；若输入非法则原样返回。
    """
    if image is None or image.size == 0:
        return image

    clip_limit = float(clip_limit)
    tile_size = max(2, int(tile_size))
    denoise_h = max(1, float(denoise_h))
    # 与 OpenCV 示例一致的搜索窗，一般无需改
    tw, sw = 7, 21

    if image.ndim == 2:
        denoised = cv.fastNlMeansDenoising(image, None, denoise_h, tw, sw)
        gray = denoised
    elif image.ndim == 3 and image.shape[2] >= 3:
        denoised = cv.fastNlMeansDenoisingColored(image, None, denoise_h, denoise_h, tw, sw)
        gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)
    else:
        return image

    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = clahe.apply(gray)

    if image.ndim == 2:
        return out
    return cv.cvtColor(out, cv.COLOR_GRAY2BGR)



image = cv.imread("Image/transfer/Image_20260407111553801.bmp")
image = image_preprocess(image, clip_limit=2.0, tile_size=8, denoise_h=5)
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.resizeWindow("image", 1000, 1000)
cv.imshow("image", image)
cv.waitKey(0)
cv.destroyAllWindows()