import pyimgsaliency as psal
import glob
import cv2
import os
import pickle
from threading import Thread
# from numba import jit

num_train_imgs = 1281167
num_val_imgs = 50000
train_input_dir = "D:/ILSVRC2012/train/"
val_input_dir = "D:/ILSVRC2012/validation/"
# train_output_dir = train_input_dir[:-1] + "_sali/"
# val_output_dir = val_input_dir[:-1] + "_sali/"
train_input_names = glob.glob(train_input_dir + '**/*.jpeg', recursive=True)
val_input_names = glob.glob(val_input_dir + '**/*.jpeg', recursive=True)
train_output_dir = train_input_dir[:-1] + "_sali_cv2/"
val_output_dir = val_input_dir[:-1] + "_sali_cv2/"


# Step 1, 2, 3, 4 of main; implemented as a method for the purpose of multithreading
# @jit
def add_saliency_cv2(train=True):
    input_names = train_input_names if train else val_input_names
    while input_names:
        current_job_filename = input_names.pop()
        try:
            if os.path.isfile(current_job_filename):
                subfolder_basename_index = current_job_filename.find('\\')
                subfolder_basename_dir = current_job_filename[subfolder_basename_index:]

                output_dir = (train_output_dir if train else val_output_dir) + subfolder_basename_dir[1:]
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)

                if os.path.isfile(output_dir):
                    print('File already exists')
                else:
                    saliency_map = calculate_saliency(current_job_filename)
                    cv2.imwrite(output_dir, saliency_map)
                    print('Successfully saved the saliency map for:', current_job_filename)
        except:
            with open('D:/ILSVRC2012/saliency_adder_cv2_log.txt', 'a') as f:
                f.write(current_job_filename + '\n')
                print('Error while handling the image. Filename added to the log.')
            f.close()


def add_saliency_mbd(train=True):
    input_names = train_input_names if train else val_input_names
    while input_names:
        current_job_filename = input_names.pop()
        try:
            if os.path.isfile(current_job_filename):
                subfolder_basename_index = current_job_filename.find('\\')
                subfolder_basename_dir = current_job_filename[subfolder_basename_index:]

                output_dir = (train_output_dir if train else val_output_dir) + subfolder_basename_dir[1:]
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)

                # rgb = cv2.imread(infile_name, cv2.IMREAD_UNCHANGED)
                if os.path.isfile(output_dir):
                    print('File already exists')
                else:
                    mbd = psal.get_saliency_mbd(current_job_filename).astype('uint8')
                    encode_mbd_to_pickle(output_dir, mbd)
        except:
            with open('D:/ILSVRC2012/saliency_adder_log.txt', 'a') as f:
                f.write(current_job_filename + '\n')
                print('Error while handling the image. Filename added to the log.')
            f.close()


def calculate_saliency(img_dir):
    sal_obj = cv2.saliency.StaticSaliencyFineGrained_create()
    image = cv2.imread(img_dir)

    (success, saliencyMap) = sal_obj.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap


def get_progressed_input_names(train=True):
    input_names = train_input_names if train else val_input_names
    output_dir = train_output_dir if train else val_output_dir

    progressed = glob.glob(output_dir + '**/*.pickle', recursive=True)
    progressed = [elm.replace('_sali', '')[:-6] + 'JPEG' for elm in progressed]
    progressed_input_names = set(input_names) - set(progressed)
    return progressed_input_names


def encode_mbd_to_pickle(output_dir, mbd):
    # output_directory.jpeg -> output_directory.pickle
    filename = output_dir[:-5] + '.pickle'
    with open(filename, 'wb') as f_out:
        pickle.dump(mbd, f_out)
    f_out.close()
    print('Successfully serialized saliency map (mbd) to:', filename)


def encode_rgbs_to_pickle(output_dir, rgb, mbd):
    # output_directory.jpeg -> output_directory.pickle
    filename = output_dir[:-5] + '.pickle'
    with open(filename, 'wb') as f_out:
        b, g, r = cv2.split(rgb)
        rgbs = cv2.merge((r, g, b, mbd))
        pickle.dump(rgbs, f_out)
    f_out.close()
    return filename


def decode_from_pickle(filename):
    rgbs = None
    with open(filename, 'rb') as f_in:
        rgbs = pickle.load(f_in)
        f_in.close()
    return rgbs


# For each subfolders (or an element of a synset),
#   1. Create a copy with the same folder hierarchy
#   2. Calculate saliency channels of the images in them
#   3. Add the saliency as the 4th channel of the images
#   4. Serialize the image with the added 4th channel as .pickle
# Result: (num_train_imgs + num_val_imgs) imgs with the 4th channel added (parsed to .pickle), copied to output_dirs
# @jit
def main():
    global train_input_names
    global val_input_names

    assert len(train_input_names) == num_train_imgs
    assert len(val_input_names) == num_val_imgs
    print("Initial assertion successful: Loaded all image directories")

    # train_input_names = get_progressed_input_names(train=True)
    # print(f"Loaded progress: {len(train_input_names)} train images left (Excluded all processed images)")
    # val_input_names = get_progressed_input_names(train=False)
    # print(f"Loaded proress: {len(val_input_names)} validation images left (Excluded all processed images)")

    train_thrd = 50
    val_thrd = 2
    train = [Thread(target=add_saliency_cv2, args=(True,)) for _ in range(train_thrd)]
    validate = [Thread(target=add_saliency_cv2, args=(False,)) for _ in range(val_thrd)]
    print(f"Created {train_thrd} threads for train dataset")
    print(f"Created {val_thrd} threads for train dataset")

    [t_thrd.start() for t_thrd in train]
    [v_thrd.start() for v_thrd in validate]
    [t_thrd.join() for t_thrd in train]
    [v_thrd.join() for v_thrd in validate]

    # add_saliency_mbd(False)
    # add_saliency_mbd(True)

    ## Final check
    train_output_names = glob.glob(train_output_dir + '**/*.jpeg', recursive=True)
    val_output_names = glob.glob(val_output_dir + '**/*.jpeg', recursive=True)
    print('===============================================================')
    print('Converted train images:', len(train_output_names), 'All converted:', len(train_output_names) == num_train_imgs)
    print('Converted validation images:', len(val_output_names), 'All converted:', len(val_output_names) == num_val_imgs)


if __name__ == '__main__':
    main()