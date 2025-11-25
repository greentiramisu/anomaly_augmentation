import argparse
import torch
import os
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.utils import save_image
from tqdm import tqdm
from perlin import rand_perlin_2d_np
import cv2
import numpy as np
from utils import realRandAugmenter, pseudoRandAugmenter, to_np_img, to_np_mask, check_inside
import glob


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

    
def augment_image(anomaly_source_img, fg_mask, normal_image, cnt, save_path_image, save_path_mask):
    #image: np.array, with float32. 

    if anomaly_source_img.dtype == np.float32 or anomaly_source_img.dtype == np.float64:
        anomaly_source_img = (anomaly_source_img * 255.0).astype(np.uint8)
    else:
        anomaly_source_img = anomaly_source_img.astype(np.uint8) # 이미 uint8인 경우


    aug = pseudoRandAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0

    fg_mask = fg_mask
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(fg_mask.shape[1], fg_mask.shape[0]))
    anomaly_img_augmented = aug(image=anomaly_source_img)

    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((256, 256), (perlin_scalex, perlin_scaley))
    threshold = 0.2
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    target_dsize = (fg_mask.shape[1], fg_mask.shape[0]) # (너비, 높이)
    resized_float_mask = cv2.resize(perlin_thr, 
                                    dsize=target_dsize, 
                                    interpolation=cv2.INTER_AREA) # INTER_LINEAR도 가능

    perlin_thr = np.where(resized_float_mask > threshold, 1.0, 0.0)

    anomaly_mask = perlin_thr * fg_mask
    anomaly_mask = np.expand_dims(anomaly_mask, axis=-1)
    img_thr = anomaly_mask * anomaly_img_augmented
    img_thr /= 255.0

    # img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    beta = torch.rand(1).numpy()[0] * 0.8


    
    perlin_thr_expanded = np.expand_dims(perlin_thr, axis=-1)
    augmented_image = normal_image * (1 - anomaly_mask) + (1 - beta) * img_thr + beta * normal_image
    
    image_tensor = torch.from_numpy(augmented_image).permute(2, 0, 1)
    mask_tensor = torch.from_numpy(anomaly_mask).permute(2,0,1)

    image_save_file = os.path.join(save_path_image, f"{cnt}.jpg")
    mask_save_file = os.path.join(save_path_mask, f"{cnt}.jpg")

    save_image(image_tensor, image_save_file)
    save_image(mask_tensor, mask_save_file)

    return augmented_image, anomaly_mask




def sample_normal_images(image_dir_path, num):
    # Normal image and fg mask obtaining.
    image_files = [f for f in os.listdir(image_dir_path) 
            if f.lower().endswith(('.jpg', '.png'))]
    image_files.sort()

    sampled_idx = random.choices(range(len(image_files)), k=num)
    sampled_files = [image_files[i] for i in sampled_idx]


    normal_path_list = []
    transform = T.ToTensor()
    for file_name in sampled_files:
        image_path = os.path.join(image_dir_path, file_name)
        normal_path_list.append(image_path)

    return normal_path_list





def sample_images(image_dir_path, num, fg_mask_path=None):
    if fg_mask_path is None:
        image_files = [f for f in os.listdir(image_dir_path) 
                    if f.lower().endswith(('.jpg', '.png'))]
        image_files.sort()
        sampled_files = image_files[:num]
        tensor_list =[]
        transform = T.ToTensor()
        for file_name in sampled_files:
            image_path = os.path.join(image_dir_path, file_name)
            try:
                image = Image.open(image_path).convert('RGB') 
                image_tensor = transform(image)
                tensor_list.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
        if not tensor_list:
            print(f"Warning: No images were successfully loaded from {image_dir_path}")
            return torch.empty(0)
        final_tensor = torch.stack(tensor_list)
        return final_tensor

    else:
        # Normal image and fg mask obtaining.
        image_files = [f for f in os.listdir(image_dir_path) 
                if f.lower().endswith(('.jpg', '.png'))]
        image_files.sort()
        fg_files = [f for f in os.listdir(fg_mask_path) 
            if f.lower().endswith(('.jpg', '.png'))]
        fg_files.sort()


        sampled_idx = random.choices(range(len(image_files)), k=num)
        sampled_files = [image_files[i] for i in sampled_idx]
        sampled_fgs = [fg_files[i] for i in sampled_idx]


        normal_path_list, fg_mask_path_list =[],[]
        transform = T.ToTensor()
        for file_name, fg_name in zip(sampled_files, sampled_fgs):
            image_path = os.path.join(image_dir_path, file_name)
            fg_path = os.path.join(fg_mask_path, fg_name)
            normal_path_list.append(image_path), fg_mask_path_list.append(fg_path)

        return normal_path_list, fg_mask_path_list


def crop_and_paste_and_save(normal_image, anomaly_image, mask_image,
                            cnt, save_path_image, save_path_mask):
    normal_part = normal_image * (1.0 - mask_image)
    anomaly_part = anomaly_image * mask_image
    new_image = normal_part + anomaly_part

    image_save_file = os.path.join(save_path_image, f"{cnt}.jpg")
    mask_save_file = os.path.join(save_path_mask, f"{cnt}.jpg")
    
    save_image(new_image, image_save_file)
    save_image(mask_image, mask_save_file)




def extended_anomalies(normal_image, anomaly_image, mask_image, fg_mask,
                        cnt, save_path_image, save_path_mask):
    #1. Augment anomaly image
    aug = realRandAugmenter()
    img_np  = to_np_img(anomaly_image)                 # HWC, uint8
    mask_np = to_np_mask(mask_image, img_np.shape[:2]) # HW,  uint8


    fg_mask = np.asarray(fg_mask).transpose(1,2,0) * 255
    fg_mask = fg_mask.astype(np.uint8)

    normal_image = np.asarray(normal_image).transpose(1,2,0) * 255
    normal_image = normal_image.astype(np.uint8)


    augmentated = aug(image=img_np, mask=mask_np)
    aug_image, aug_mask = augmentated['image'], augmentated['mask'] 
    # aug_image.shape= (1000, 1500, 3), aug_mask.shape = (1000, 1500, 3)

    n_image, aug_mask_shifted = check_inside(aug_image, aug_mask, normal_image.copy(), fg_mask)
    beta = np.random.uniform(low=0.0, high=0.8)
    aug_mask_3ch = np.stack((aug_mask_shifted,) * 3, axis=-1) / 255


    normal_image_float = normal_image.astype(np.float32) / 255.0
    n_image_float = n_image.astype(np.float32) / 255.0

    # 3. (★수정★) 모든 계산을 float(0.0-1.0) 공간에서 수행
    blend_color = (1.0 - beta) * n_image_float + beta * normal_image_float
    obtained_image_float = normal_image_float * (1 - aug_mask_3ch) + aug_mask_3ch * blend_color    

    mask_tensor = torch.from_numpy(aug_mask_shifted / 255.0).float()
    mask_tensor = mask_tensor.unsqueeze(0) # (H, W) -> (1, H, W)

    image_tensor = torch.from_numpy(obtained_image_float)
    image_tensor = image_tensor.permute(2, 0, 1).float()

    image_save_file = os.path.join(save_path_image, f"{cnt}.jpg")
    mask_save_file = os.path.join(save_path_mask, f"{cnt}.jpg")

    save_image(image_tensor, image_save_file)
    save_image(mask_tensor, mask_save_file)







def sample_dtd(path):
    file_list = glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.jpeg")
    random_file_path = random.choice(file_list)
    image_pil = Image.open(random_file_path)
    dtd_np = np.array(image_pil)
    return dtd_np




def shuffle_image_grid(image: np.ndarray, grid_shape: tuple = (8, 8)) -> np.ndarray:
    
    h, w, *c = image.shape
    grid_h, grid_w = grid_shape
    
    new_h = (h // grid_h) * grid_h
    new_w = (w // grid_w) * grid_w
    image_cropped = image[:new_h, :new_w]
    
    h_strips = np.split(image_cropped, grid_h, axis=0)
    
    tiles_list = []
    for strip in h_strips:
        tiles_in_strip = np.split(strip, grid_w, axis=1)
        tiles_list.extend(tiles_in_strip)
            
    random.shuffle(tiles_list)
    
    rows = []
    for i in range(grid_h):
        start_idx = i * grid_w
        end_idx = (i + 1) * grid_w
        row = np.concatenate(tiles_list[start_idx:end_idx], axis=1)
        rows.append(row)

    shuffled_image = np.concatenate(rows, axis=0)
    
    return shuffled_image



def simulated_anomalies(normal_image, fg_mask, dtd_folder_path,
                        cnt, save_path_image, save_path_mask):
    
    normal_image, fg_mask = np.asarray(normal_image).transpose(1,2,0), np.asarray(fg_mask).transpose(1,2,0)[:, :, 0]
    if random.random() >= 0.5:
        dtd_np = sample_dtd(dtd_folder_path)
        augmented_image, anomaly_mask = augment_image(dtd_np, fg_mask, normal_image,
                                                      cnt, save_path_image, save_path_mask)
    else:
        normal_aug_image = shuffle_image_grid(normal_image)
        augmented_image, anomaly_mask = augment_image(normal_aug_image, fg_mask, normal_image,
                                                        cnt, save_path_image, save_path_mask)
        






def main_prnet(data_root,
                object, 
                anomaly, 
                num_images, 
                gen_path,
                fg_mask_root_path,
                dtd_path):
    
    anomaly_dir_path = os.path.join(data_root, object, 'test', anomaly)
    mask_dir_path = os.path.join(data_root, object, 'ground_truth', anomaly)
    normal_dir_path = os.path.join(data_root, object, 'train', 'good')
    fg_mask_path = os.path.join(fg_mask_root_path, object, 'train/masks')

    if '3d' in data_root.lower():
        anomaly_dir_path = os.path.join(data_root, object, 'test', anomaly, 'rgb')
        mask_dir_path = os.path.join(data_root, object, 'test', anomaly, 'gt')
        normal_dir_path = os.path.join(data_root, object, 'train', 'good', 'rgb')
    image_files = [f for f in os.listdir(anomaly_dir_path) 
                       if f.lower().endswith(('.jpg', '.png'))]
    num_to_sample = len(image_files)//3 if 'dagm' not in data_root.lower() else 5

    if num_to_sample == 0:
        print(f"Warning: Not enough images to sample 1/3 in {anomaly_dir_path}. (Total: {len(image_files)})")
        return torch.empty(0)
    
    anomaly_images = sample_images(anomaly_dir_path, num_to_sample)
    mask_images = sample_images(mask_dir_path, num_to_sample)

    normal_num_per_anomaly = num_images // num_to_sample 
    cnt=0

    save_path_image = os.path.join(gen_path, object, anomaly, 'image')
    save_path_mask = os.path.join(gen_path, object, anomaly, 'mask')

    os.makedirs(save_path_image, exist_ok=True)
    os.makedirs(save_path_mask, exist_ok=True)
    transform = T.ToTensor()

    with tqdm(total=num_images, desc=f"Generating {object}-{anomaly}, anomaly_images: {num_to_sample}") as pbar:
        for (anomaly_image, mask_image) in zip(anomaly_images, mask_images):
            normal_path_list, fg_mask_path_list = sample_images(normal_dir_path, normal_num_per_anomaly, fg_mask_path)
            for normal_path, fg_mask_file_path in zip(normal_path_list, fg_mask_path_list):
                if cnt >= num_images:
                    break

                
                image, fg = Image.open(normal_path).convert('RGB'), Image.open(fg_mask_file_path).convert('RGB')
                normal_image, fg_mask = transform(image), transform(fg)

                # Algorithm 
                if random.random() >= 0.5:
                    extended_anomalies(normal_image, anomaly_image, mask_image, fg_mask, 
                                       cnt, save_path_image, save_path_mask)
                else:
                    simulated_anomalies(normal_image, mask_image, dtd_path,
                                        cnt, save_path_image, save_path_mask)
                    

                cnt += 1
                pbar.update(1) 

            if cnt >= num_images:
                break
        

        while cnt < num_images: 

            image, fg = Image.open(normal_path_list[0]).convert('RGB'), Image.open(fg_mask_path_list[0]).convert('RGB')
            normal_image, fg_mask = transform(image), transform(fg)

            extended_anomalies(normal_image, anomaly_images[0], mask_images[0], fg_mask,
                                cnt, save_path_image, save_path_mask)
            cnt += 1
            pbar.update(1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument('--gen_path', type=str, default='./CaP/')
    parser.add_argument('--fg_mask_path', type=str, default='./fg_mask/')
    parser.add_argument('--dtd_path', default='./dtd')
    args = parser.parse_args()

    objects = os.listdir(args.data_root)
    os.makedirs(args.gen_path, exist_ok=True)
    
    for object in objects:
        test_path = os.path.join(args.data_root, object, 'test')
        anomalies = [d for d in os.listdir(test_path) if d != 'good']
        for anomaly in anomalies:
            print(f"Image Generating... {object} - {anomaly}")
            if 'dagm' not in args.data_root.lower():
                main_prnet(data_root=args.data_root, 
                            object=object,
                            anomaly=anomaly,
                            num_images = args.num_images,
                            gen_path = args.gen_path,
                            fg_mask_root_path = args.fg_mask_path,
                            dtd_path=args.dtd_path)

