import torch
from helpers import join, join_and_create_dir, files, chunks
from prepare_dataset import parse_flickr_labels, prepare_web_logos
import cv2
import numpy as np

def get_one_image(img_list):
    max_width = 0
    total_height = 200  # padding
    for img in img_list:
        if img.shape[1] > max_width:
            max_width = img.shape[1]
        total_height += img.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in img_list:
        # add an image to the final array and increment the y coordinate
        image = np.hstack((image, np.zeros((image.shape[0], max_width - image.shape[1], 3))))
        final_image[current_y:current_y + image.shape[0], :, :] = image
        current_y += image.shape[0]
    return final_image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp10/weights/best.pt')  # default

img_paths = files("./datasets/custom_ds/val/images/")

out_folder = join_and_create_dir("./results_val")

for img_paths in chunks(img_paths, 6):
    imgs = [cv2.imread(x) for x in img_paths]
    imgs = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in imgs]
    
    # Inference
    results = model(img_paths)

    # Results
    results.print()
    results.save(out_folder)

    results.xyxy[0]  # img1 predictions (tensor)
    a = results.pandas().xyxy  # img1 predictions (pandas)
    print(a)
    #      xmin    ymin    xmax   ymax  confidence  class    name
    # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie


img_results_paths = files(out_folder)
combined_val = join_and_create_dir("./combined_val")

for i, img_paths in enumerate(chunks(img_results_paths, 6)):
    imgs = [cv2.imread(x) for x in img_paths]


    combined = get_one_image(imgs)
    cv2.imwrite(f"{combined_val}/{i}.png", combined)



