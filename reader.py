import csv
import cfg
import os
import cv2
import numpy as np
from keras import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
import random



def should_skip_zero_steering_item():
    """from X items where (steering == 0) skip Y items to fight unbalanced input"""
    cfg.state.ignore_zero_steer_items_count += 1
    if cfg.state.ignore_zero_steer_items_count > cfg.ignore_zero_steer_items_from:
        cfg.state.ignore_zero_steer_items_count = 1
    return cfg.state.ignore_zero_steer_items_count <= cfg.ignore_zero_steer_items_skip


def get_all_meta(first_dataset, last_dataset, check_files_exist=True, verbose=False):
    """
    Reads in all image paths and steering values from given folders (last exclusive).
    Schema: steering (float), img path (str).
    """
    meta_db = []
    for data_set in range(first_dataset, last_dataset):
        path = cfg.data_root_path_fmt.format(data_set) + cfg.csv_rel_path
        print('Reading file {}...'.format(path), end='')
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            count_possible, count_skipped_straight, count_ok = 0, 0, 0
            for line in reader:
                center_img, left_img, right_img, angle, _, _, _ = line
                if angle == 0 and should_skip_zero_steering_item():
                    count_skipped_straight += 1
                    continue
                angle = float(angle)
                count_possible += 3
                # center image
                if (not check_files_exist) or (os.path.exists(center_img)):
                    meta_db.append([line[0], angle])
                    count_ok += 1
                elif verbose:
                    print('File missing:', line[0])
                # left image
                if (not check_files_exist) or (os.path.exists(left_img)):
                    meta_db.append([line[1], angle * cfg.camera_steer_multiplier + cfg.camera_steer_offset])
                    count_ok += 1
                elif verbose:
                    print('File missing:', line[1])
                # right image
                if (not check_files_exist) or (os.path.exists(right_img)):
                    meta_db.append([line[2], angle * cfg.camera_steer_multiplier - cfg.camera_steer_offset])
                    count_ok += 1
                elif verbose:
                    print('File missing:', line[2])
            print('done. From {} skipped {}, read {} items successfully.{}'.format(count_possible,
                count_skipped_straight, count_ok, ' Images checked for existence.' if check_files_exist else ''))
    print('Total valid items explored:', len(meta_db))
    return meta_db


def show_examples(X, y, lines, columns):
    fig, ax = plt.subplots(lines, columns, figsize=(18, 10))
    for vert in range(lines):
        for horiz in range(columns):
            i = random.randrange(len(X))
            ax[vert][horiz].imshow(X[i])
            ax[vert][horiz].title.set_text(y[i])
    plt.tight_layout()
    plt.show()


def generator(meta_db, batch_size):
    """Returns data for keras.fit in form of a list of (X_Train, y_train)"""
    num_samples = len(meta_db)
    batch_mod_size = batch_size // cfg.generator_new_item_multiplier
    if cfg.enable_datagen:
        batch_mod_size = batch_mod_size // cfg.datagen_item_multiplier

    datagen = preprocessing.image.ImageDataGenerator(
        rotation_range=cfg.datagen_rotation_range,
        #height_shift_range=0.02,
        #brightness_range=[0.9, 1.1],
        dtype=tf.uint8,
    )
    while 1:
        for offset in range(0, num_samples, batch_mod_size):
            batch_metas = meta_db[offset : offset + batch_mod_size]
            images, angles = [], []
            for path, angle in batch_metas:
                # generate orig
                image = cv2.imread(path)
                image = image[:,:,::-1]
                images.append(image)
                angles.append(angle)
                # generate flipped
                flipped = image[:,::-1,:]
                images.append(flipped)
                angles.append(-1.0 * angle)
                # generate randomized
                if cfg.enable_datagen:
                    rnd = datagen.random_transform(image)
                    images.append(rnd)
                    angles.append(angle)
                    # generate flipped randomized
                    rnd_flipped = datagen.random_transform(flipped)
                    images.append(rnd_flipped)
                    angles.append(-1.0 * angle)

            X = np.array(images)
            y = np.array(angles)

            if cfg.debug_show_example_images:
                if cfg.state.debug_show_example_images_count < cfg.debug_show_example_images:
                    show_examples(X, y, lines=2, columns=4)
                    cfg.state.debug_show_example_images_count += 1

            yield (X, y)


if __name__ == '__main__':
    meta_db = get_all_meta(1, 5, check_files_exist=True, verbose=False)
