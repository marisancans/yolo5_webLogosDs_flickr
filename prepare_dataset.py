from helpers import read_file, dirs, files, basename, create_dir, join, join_and_create_dir, cp, write_file, group_buckets, stem
import cv2
import numpy as np

import xml.etree.ElementTree as ET

def label_to_idx(labels, name):
    for i, k in enumerate(labels):
        if k == name:
            return i

    return None


def proportion_bucketed(items, percent):
    to = int(len(items) * percent)

    keys = list(items.keys())

    train = {k: items[k] for k in keys[0:to]}
    val = {k: items[k] for k in keys[to:len(items)]}
    return train, val


def proportion(items, percent):
    to = int(len(items) * percent)
    e_train = items[0:to]
    e_test = items[to:len(items)]
    return e_train, e_test




def parse_flickr_labels(label_path):
    lines = read_file(label_path)

    flick_data = []

    for line in lines:
        filename, label, train_subset, x1, y1, x2, y2 = line.strip().split(" ")
        flick_data.append({
            "filename": filename, 
            "label": label.lower(), 
            "train_subset": train_subset, 
            "x1": float(x1), 
            "y1": float(y1), 
            "x2": float(x2), 
            "y2": float(y2),
        })

    # Distinct labels
    labels = set([x["label"] for x in flick_data])
    labels = sorted(labels)
    print(labels)
    print(len(labels))

    return flick_data, labels


def format_yolo(label_buckets, flickr_path, images_dir, labels_dir, labels):
    for filename, label_bucket in label_buckets.items():
        # get rid of subsets
        subset_bucket = group_buckets(label_bucket, key = lambda x : x["train_subset"])

        # safety check - make sure all subsets have equal length, if not, this dataset has errors?
        lens_subset = [len(x) for x in subset_bucket.values()]
        lens_max_id = np.argmax(lens_subset)

        if len(set(lens_subset)) > 1:
            print("Image contains more than one class")

        # safety check - make sure all subsets contain the same data - its possible that subsets have splited labels
        subset_vals = []
        for subset in subset_bucket.values():
            dumb_compare = []

            for l in subset:
                dumb_str = f"{l['x1']}_{l['y1']}_{l['x2']}_{l['y2']}"
                dumb_compare.append(dumb_str)

            val = "X".join(dumb_compare)
            subset_vals.append(val)

        # if len(set(subset_vals)) > 1:
            # Exception("Something wrong with subset equality")

        single_subset = list(subset_bucket.values())[lens_max_id]

        img_src_path = join(flickr_path, "flickr_logos_27_dataset_images", filename)

        if len(single_subset) > 1:
            print(f"{filename} contains more than one bbox")
           
        lines = []

        img = cv2.imread(img_src_path)
        h, w, c = img.shape

        img_out_path = images_dir + f"/{filename}"
        cv2.imwrite(img_out_path, img)
        
        # each item is a bbox
        for item in single_subset:
            label_idx = label_to_idx(labels, item["label"])
                    
            # x_center y_center width height
            x1_bbox = item["x1"]
            y1_bbox = item["y1"]
            x2_bbox = item["x2"]
            y2_bbox = item["y2"]

            w_bbox = x2_bbox - x1_bbox
            h_bbox = y2_bbox - y1_bbox

            x = x1_bbox + w_bbox / 2
            y = y1_bbox + h_bbox / 2

            x_norm = x / w
            y_norm = y / h

            w_bbox_norm = w_bbox / w
            h_bbox_norm = h_bbox / h

            line = f"{label_idx} {x_norm} {y_norm} {w_bbox_norm} {h_bbox_norm}\n"
            lines.append(line)

        st = stem(filename) + ".txt"
        path_lines = join(labels_dir, st)
        write_file(path_lines, lines)


def prepare_flickr(union_labels, flick_data, dataset_path, flickr_path, train_proportion):

    # make sure we handle cases where multiple labels exist per image
    label_buckets = group_buckets(flick_data, key = lambda x: x["filename"])
    print(len(flick_data), len(label_buckets))

    train, val = proportion_bucketed(label_buckets, train_proportion)
    # split the dataset into train and validation sets


    for ds, mode in zip([train, val], ["train", "val"]):    
        images_dir_ds = join_and_create_dir(dataset_path, mode, "images",)
        labels_dir_ds = join_and_create_dir(dataset_path, mode, "labels")
        format_yolo(ds, flickr_path, images_dir_ds, labels_dir_ds, union_labels)    






def parse_web_logos_labels(web_logo_path):
    anotation_folder = join(web_logo_path, "Annotations")
    annoation_paths = files(anotation_folder)

    web_logo_data = []

    for annoation_path in annoation_paths:
        tree = ET.parse(annoation_path)
        root = tree.getroot()

        filename = root.find('filename').text
        width = root.find('size').find('width').text
        height = root.find('size').find('height').text

        object = root.find('object')
        name = object.find('name').text
        bndbox = object.find('bndbox')

        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        web_logo_data.append({
            "filename": filename, 
            "label": name.lower(), 
            "x1": float(xmin), 
            "y1": float(ymin), 
            "x2": float(xmax), 
            "y2": float(ymax),
            "width": float(width),
            "height": float(height)
        })

    # Distinct labels
    labels = set([x["label"] for x in web_logo_data])
    labels = sorted(labels)
    print(labels)
    print(len(labels))
    
    return web_logo_data, labels
        


def format_web_logos(data, web_logo_path, images_dir, labels_dir, labels):
    for item in data:
        lines = []

        w = item["width"]
        h = item["height"]

        filename = item["filename"]

        label_idx = label_to_idx(labels, item["label"])

        # x_center y_center width height
        x1_bbox = item["x1"]
        y1_bbox = item["y1"]
        x2_bbox = item["x2"]
        y2_bbox = item["y2"]

        w_bbox = x2_bbox - x1_bbox
        h_bbox = y2_bbox - y1_bbox

        x = x1_bbox + w_bbox / 2
        y = y1_bbox + h_bbox / 2

        x_norm = x / w
        y_norm = y / h

        w_bbox_norm = w_bbox / w
        h_bbox_norm = h_bbox / h
        
        line = f"{label_idx} {x_norm} {y_norm} {w_bbox_norm} {h_bbox_norm}\n"
        lines.append(line)

        st = stem(filename) + ".txt"
        path_lines = join(labels_dir, st)

        write_file(path_lines, lines)


        img_src = join(web_logo_path, "JPEGImages", filename)
        cp(img_src, images_dir)




def prepare_web_logos(union_labels, web_logo_data, dataset_path, web_logo_path, train_proportion): 
    train, val = proportion(web_logo_data, train_proportion)
    # split the dataset into train and validation sets


    for ds, mode in zip([train, val], ["train", "val"]):    
        images_dir_ds = join_and_create_dir(dataset_path, mode, "images",)
        labels_dir_ds = join_and_create_dir(dataset_path, mode, "labels")
        format_web_logos(ds, web_logo_path, images_dir_ds, labels_dir_ds, union_labels)   


def main():
    flickr_path = "./flickr_logos_27_dataset"
    flick_annotation_path = join(flickr_path, "flickr_logos_27_dataset_training_set_annotation.txt")

    dataset_path = "./datasets/custom_ds"

    flickr_data, flickr_labels = parse_flickr_labels(flick_annotation_path)

    
    web_logo_path = "./Evaluation_Dataset"
    web_logo_data, web_logo_labels = parse_web_logos_labels(web_logo_path)

    intersect_labels = list(set(flickr_labels) & set(web_logo_labels))
    intersect_labels = sorted(intersect_labels)
    print(intersect_labels)

    # were interested in union of both dataset labels    
    union_labels = set.union(set(flickr_labels), set(web_logo_labels))

    # filter out only aviable classes
    flickr_data = [x for x in flickr_data if x["label"] in intersect_labels]
    web_logo_data = [x for x in web_logo_data if x["label"] in intersect_labels]

    train_proportion = 0.7
    prepare_flickr(intersect_labels, flickr_data, dataset_path, flickr_path, train_proportion) # 566


    prepare_web_logos(intersect_labels, web_logo_data, dataset_path, web_logo_path, train_proportion) # 5077


if __name__ == "__main__":
    main()






