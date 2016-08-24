import dicom, glob, os, subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import ndimage

def generate_imgs(folder, threshold=0.1):
    images = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

    fig = plt.figure()
    compound_1 = np.zeros(images[0].pixel_array.shape, dtype=bool)
    compound_2 = np.zeros(images[0].pixel_array.shape, dtype=bool)

    for i in xrange(len(images) - 1):
        img_prev = images[i].pixel_array / 256.
        img_next = images[i + 1].pixel_array / 256.
        diff1 = abs(img_next - img_prev) > threshold
        diff2 = ndimage.binary_erosion(diff1)
        compound_1 |= diff1
        compound_2 |= diff2

        fig.add_subplot(2, 3, 1)
        plt.imshow(img_prev, cmap=cm.Greys_r)
        plt.axis('off')

        fig.add_subplot(2, 3, 2)
        plt.imshow(diff1, cmap=cm.Greys_r)
        plt.axis('off')

        fig.add_subplot(2, 3, 3)
        plt.imshow(diff2, cmap=cm.Greys_r)
        plt.axis('off')

        # another line
        fig.add_subplot(2, 3, 5)
        plt.imshow(compound_1, cmap=cm.Greys_r)
        plt.axis('off')

        fig.add_subplot(2, 3, 6)
        plt.imshow(compound_2, cmap=cm.Greys_r)
        plt.axis('off')

        plt.savefig("videos/tmp_file%02d.png" % i)

def generate_video(folder):
    os.chdir('videos')

    tmp = folder.split('/')
    el1, el2 = int(tmp[2]), tmp[4]
    file_name = "_video_%03d_%s.mp4" % (el1, el2)
    if os.path.exists(file_name):
        os.remove(file_name)

    command = 'ffmpeg -framerate 5 -i tmp_file%02d.png -r 30 -pix_fmt yuv420p ' + file_name
    subprocess.call(command.split(' '))
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    os.chdir('..')

def generate_many_videos():
    paths = glob.glob('data/train/*/')
    for i in paths[0:10]:
        for location in glob.glob(i + 'study/sax_*'):
            folder = location + '/'
            generate_imgs(folder)
            generate_video(folder)

generate_many_videos()
