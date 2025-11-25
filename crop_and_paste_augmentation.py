import argparse
import torch
import os
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.utils import save_image
from tqdm import tqdm


def sample_images(image_dir_path, num, random_sample=False):
    image_files = [f for f in os.listdir(image_dir_path) 
                    if f.lower().endswith(('.jpg', '.png'))]
    image_files.sort()
    sampled_files = image_files[:num] if random_sample == False else random.sample(image_files, num)

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




def crop_and_paste_and_save(normal_image, anomaly_image, mask_image,
                            cnt, save_path_image, save_path_mask):
    normal_part = normal_image * (1.0 - mask_image)
    anomaly_part = anomaly_image * mask_image
    new_image = normal_part + anomaly_part

    image_save_file = os.path.join(save_path_image, f"{cnt}.jpg")
    mask_save_file = os.path.join(save_path_mask, f"{cnt}.jpg")
    
    save_image(new_image, image_save_file)
    save_image(mask_image, mask_save_file)





def main_crop_and_paste(data_root,
                        object, 
                        anomaly, 
                        num_images, 
                        gen_path):
    
    anomaly_dir_path = os.path.join(data_root, object, 'test', anomaly)
    mask_dir_path = os.path.join(data_root, object, 'ground_truth', anomaly)
    normal_dir_path = os.path.join(data_root, object, 'train', 'good')

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


    with tqdm(total=num_images, desc=f"Generating {object}-{anomaly}, anomaly_images: {num_to_sample}") as pbar:
        for (anomaly_image, mask_image) in zip(anomaly_images, mask_images):

            normal_images = sample_images(normal_dir_path, normal_num_per_anomaly)

            for normal_image in normal_images:
                if cnt >= num_images:
                    break
                crop_and_paste_and_save(normal_image, anomaly_image, mask_image,
                                        cnt, save_path_image, save_path_mask)
                cnt += 1
                pbar.update(1) 

            if cnt >= num_images:
                break
        

        while cnt < num_images: 
            if normal_images.shape[0] == 0: 
                 normal_images = sample_images(normal_dir_path, 1)
                 
                 if normal_images.shape[0] == 0:
                     pbar.write(f"Error: No normal images found for {object}. Stopping.")
                     break

            crop_and_paste_and_save(normal_images[0], anomaly_images[0], mask_images[0],
                                    cnt, save_path_image, save_path_mask)
            cnt += 1
            pbar.update(1)

            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument('--gen_path', type=str, default='./CaP/')
    args = parser.parse_args()

    objects = os.listdir(args.data_root)
    os.makedirs(args.gen_path, exist_ok=True)

    for object in objects:
        test_path = os.path.join(args.data_root, object, 'test')
        anomalies = [d for d in os.listdir(test_path) if d != 'good']
        for anomaly in anomalies:
            print(f"Image Generating... {object} - {anomaly}")
            main_crop_and_paste(data_root=args.data_root, 
                           object=object,
                         anomaly=anomaly,
                         num_images = args.num_images,
                         gen_path = args.gen_path)

