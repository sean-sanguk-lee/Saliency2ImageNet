import pyimgsaliency as psal
import glob
import cv2
import os
import pickle
from threading import Thread

num_train_imgs = 1281167
num_val_imgs = 50000
train_input_dir = "D:/ILSVRC2012/train/"
val_input_dir = "D:/ILSVRC2012/validation/"
train_output_dir = train_input_dir[:-1] + "_sali/"
val_output_dir = val_input_dir[:-1] + "_sali/"
current_infile_name = ''


# Step 1, 2, 3, 4 of main; implemented as a method for the purpose of multithreading
def add_saliency_mbd(train=True):
    input_dir = (train_input_dir if train else val_input_dir) + '**/*.jpeg'
    for infile_name in glob.glob(input_dir, recursive=True):
        try:
            if not os.path.isdir(infile_name):
                subfolder_basename_index = infile_name.find('\\')
                subfolder_basename_dir = infile_name[subfolder_basename_index:]

                output_dir = (train_output_dir if train else val_output_dir) + subfolder_basename_dir[1:]
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
                current_infile_name = infile_name

                rgb = cv2.imread(infile_name, cv2.IMREAD_UNCHANGED)
                mbd = psal.get_saliency_mbd(infile_name).astype('uint8')
                final_output_dir = encode_to_pickle(output_dir, rgb, mbd)
                print('Successfully created an rgbs img to:', final_output_dir)
        except:
            with open('D:/ILSVRC2012/saliency_adder_log.txt', 'a') as f:
                f.write(current_infile_name + '\n')
                print('Error while handling the image. Filename added to the log.')
            f.close()


def encode_to_pickle(output_dir, rgb, mbd):
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
def main():
    train_input_names = glob.glob(train_input_dir + '**/*.jpeg', recursive=True)
    val_input_names = glob.glob(val_input_dir + '**/*.jpeg', recursive=True)
    assert len(train_input_names) == num_train_imgs
    assert len(val_input_names) == num_val_imgs

    train = Thread(target=add_saliency_mbd, args=(True,))
    validate = Thread(target=add_saliency_mbd, args=(False,))

    train.start()
    validate.start()
    train.join()
    validate.join()

    # add_saliency_mbd(False)
    # add_saliency_mbd(True)

    train_output_names = glob.glob(train_output_dir + '**/*.pickle', recursive=True)
    val_output_names = glob.glob(val_output_dir + '**/*.pickle', recursive=True)
    print('Converted train images:', len(train_output_dir), 'All converted:', len(train_output_names) == num_train_imgs)
    print('Converted validation images:', len(val_output_names), 'All converted:', len(val_output_names) == num_val_imgs)


if __name__ == '__main__':
    main()