import numpy as np
from perlin import rand_perlin_2d_np
import torch
import imgaug.augmenters as iaa
import albumentations as A
import cv2
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

def plot_images(*images):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 6)) # figsize는 그림 크기(인치)

    # ** 중요: 이미지가 1개일 때 axes는 배열이 아니므로 리스트로 감싸줍니다. **
    if num_images == 1:
        axes = [axes]

    # 2. 첫 번째 플롯(axes[0])에 이미지 그리기
    for i in range(num_images):
            image = images[i]
            
            # 3-1. 이미지 차원에 따라 (RGB/Mask) 다르게 표시
            if image.ndim == 2:
                # 2D 이미지 (Mask)는 흑백(gray)으로 표시
                axes[i].imshow(image, cmap='gray')
            else:
                # 3D 이미지 (RGB)
                axes[i].imshow(image)
            axes[i].axis('off')          # 축 번호 숨기기

    # 4. 그림을 화면에 보여줍니다.
    plt.tight_layout() # 그림 간격 자동 조절
    plt.show()


augmentors_for_real = [
    A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ], p=1.0),
    A.OpticalDistortion(p=1.0, distort_limit=1.0),
    A.OneOf([
        A.GaussNoise(p=1.0),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
        A.Blur(blur_limit=3, p=1.0),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=1.0),
        A.GridDistortion(p=1.0),
        A.PiecewiseAffine(p=1.0),   # IAAPiecewiseAffine -> PiecewiseAffine
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2, p=1.0),
        A.Sharpen(p=1.0),            # IAASharpen -> Sharpen
        A.Emboss(p=1.0),             # IAAEmboss -> Emboss
        A.RandomBrightnessContrast(p=1.0),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
]

augmentors_for_pseudo = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                iaa.pillike.EnhanceSharpness(),
                iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                iaa.Solarize(0.5, threshold=(32,128)),
                iaa.Posterize(),
                iaa.Invert(),
                iaa.pillike.Autocontrast(),
                iaa.pillike.Equalize(),
                iaa.Affine(rotate=(-45, 45))]


def realRandAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmentors_for_real)), 3, replace=False)
    aug = A.Compose([augmentors_for_real[aug_ind[0]],
                    augmentors_for_real[aug_ind[1]],
                    augmentors_for_real[aug_ind[2]]])
    return aug

def pseudoRandAugmenter():
    aug_ind = np.random.choice(np.arange(len(augmentors_for_pseudo)), 3, replace=False)
    aug = iaa.Sequential([augmentors_for_pseudo[aug_ind[0]],
                        augmentors_for_pseudo[aug_ind[1]],
                        augmentors_for_pseudo[aug_ind[2]]])
    return aug

import numpy as np
import torch

def to_np_img(t: torch.Tensor) -> np.ndarray:
    """
    torch.Tensor (C,H,W) 또는 (H,W) → numpy uint8 (H,W,C) 또는 (H,W,1)
    값이 float일 땐 [0,1]로 가정하고 0~255로 스케일링.
    """
    t = t.detach().cpu()
    if t.ndim == 3 and t.shape[0] in (1, 3, 4):   # [C,H,W] -> [H,W,C]
        t = t.permute(1, 2, 0)
    elif t.ndim == 2:                              # [H,W] -> [H,W,1]
        t = t.unsqueeze(-1)

    if t.dtype in (torch.float32, torch.float64):
        t = (t.clamp(0, 1) * 255).to(torch.uint8)
    else:
        t = t.to(torch.uint8)

    return np.ascontiguousarray(t.numpy())

def to_np_mask(t: torch.Tensor, target_hw=None) -> np.ndarray:
    """
    torch.Tensor (C,H,W)/(H,W) → numpy uint8 (H,W)
    float일 땐 0/1 기준으로 이진화 후 0/255로 변환.
    target_hw가 주어지면 (H,W)로 리사이즈(NEAREST).
    """
    t = t.detach().cpu()
    # 채널 제거: [C,H,W]면 첫 채널 사용
    if t.ndim == 3:
        if t.shape[0] > 1:
            t = t[0]
        else:
            t = t.squeeze(0)
    t = t.squeeze()  # [H,W]

    if t.dtype in (torch.float32, torch.float64):
        t = (t > 0.5).to(torch.uint8) * 255
    else:
        t = t.to(torch.uint8)

    m = t.numpy()
    if target_hw is not None and (m.shape[0] != target_hw[0] or m.shape[1] != target_hw[1]):
        import cv2
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

    return np.ascontiguousarray(m)



# aug_image.shape= (1000, 1500, 3), aug_mask.shape = (1000, 1500, 3)

def check_inside(aug_image, aug_mask, n_image, fg_mask):

    if aug_mask.ndim == 3:
        aug_mask_2d = (aug_mask[:, :, 0] == 255).astype(np.uint8) * 255
        # 또는 np.all(aug_mask == 255, axis=2).astype(np.uint8) * 255
    else:
        aug_mask_2d = aug_mask.astype(np.uint8) # 이미 2D라면 dtype만 보장


    fg_mask = fg_mask[:,:,0]

    img_height, img_width = n_image.shape[0], n_image.shape[1]
    #aug_mask_2d.shape = (1000, 1500)



    intersect_mask = np.logical_and(fg_mask == 255, aug_mask_2d == 255)
    if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask_2d == 255))):
        # when most part of aug_mask is in the fg_mask region 
        # copy the augmentated anomaly area to the normal image
        n_image[aug_mask == 255, :] = aug_image[aug_mask_2d == 255, :]
        return n_image, aug_mask_2d
    else:
        contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_xs, center_ys = [], []
        widths, heights = [], []
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            if M['m00'] == 0:  # error case
                x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
            else:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            center_xs.append(center_x)
            center_ys.append(center_y)
            x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
            y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
            width, height = x_max - x_min, y_max - y_min
            widths.append(width)
            heights.append(height)
        if len(widths) == 0 or len(heights) == 0:  # no contours
            n_image[aug_mask_2d == 255, :] = aug_image[aug_mask_2d == 255, :]
            return n_image, aug_mask_2d
        else:
            max_width, max_height = np.max(widths), np.max(heights)
            center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
            fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

            x_coord = np.arange(0, img_width)
            y_coord = np.arange(0, img_height)
            xx, yy = np.meshgrid(x_coord, y_coord)
            # coordinates of fg region points
            xx_fg = xx[fg_mask]
            yy_fg = yy[fg_mask]
            xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)
            
            if xx_yy_fg.shape[0] == 0:  # no fg
                n_image[aug_mask_2d == 255, :] = aug_image[aug_mask_2d == 255, :]
                return n_image, aug_mask_2d

            aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
            for i in range(len(contours)):
                aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                # random generate a point in the fg region
                idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                rand_xy = xx_yy_fg[idx]
                delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                
                x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                
                # mask for one anomaly region
                aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                aug_mask_i[y_min:y_max, x_min:x_max] = 255
                aug_mask_i = np.logical_and(aug_mask_2d == 255, aug_mask_i == 255)
                
                # coordinates of orginal mask points
                xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                
                # shift the original mask into fg region
                xx_ano_shifted = xx_ano - delta_x
                yy_ano_shifted = yy_ano - delta_y
                outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                outer_points = np.logical_or(outer_points_x, outer_points_y)
                
                # keep points in image
                xx_ano_shifted = xx_ano_shifted[~outer_points]
                yy_ano_shifted = yy_ano_shifted[~outer_points]
                aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                
                # original points should be changed
                xx_ano = xx_ano[~outer_points]
                yy_ano = yy_ano[~outer_points]
                new_aug_mask_i[yy_ano, xx_ano] = 255
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                aug_mask_shifted[aug_mask_shifted_i == 255] = 255
            return n_image, aug_mask_shifted