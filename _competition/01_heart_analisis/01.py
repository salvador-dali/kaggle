import dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
import subprocess
from skimage import feature

def show_some_images(folder):
    img = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

    fig = plt.figure()
    for i in xrange(6):
        fig.add_subplot(2, 3, i + 1)
        plt.imshow(img[5 * i].pixel_array, cmap=cm.Greys_r)

    plt.show()

def generate_video(folder):
    # ffmpeg -framerate 8 -i file%02d.png -r 30 -pix_fmt yuv420p out.mp4
    img = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]
    for i in xrange(len(img)):
        plt.imshow(img[i].pixel_array, cmap=cm.Greys_r)
        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir(folder)
    video_name = "_video_" + folder.split('/')[-2]
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        '../' + video_name + '.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

def check_edges(folder):
    arr_img = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]
    img = arr_img[0].pixel_array / 256.

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(img, cmap=cm.Greys_r)

    fig.add_subplot(1, 2, 2)
    plt.imshow(feature.canny(img, sigma=2), cmap=cm.Greys_r)

    plt.show()

def generate_video_pair(folder):
    # ffmpeg -framerate 8 -i file%02d.png -r 30 -pix_fmt yuv420p out.mp4
    arr_img = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

    for i in xrange(len(arr_img)):
        img = arr_img[i].pixel_array / 256.

        fig = plt.figure()

        fig.add_subplot(1, 2, 1)
        plt.imshow(img, cmap=cm.Greys_r)

        fig.add_subplot(1, 2, 2)
        plt.imshow(feature.canny(img, sigma=3), cmap=cm.Greys_r)

        plt.savefig(folder + "/file%02d.png" % i)

    os.chdir(folder)
    video_name = "_video_pair_" + folder.split('/')[-2]
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        '../' + video_name + '.mp4'
    ], stdout=open(os.devnull, 'w'))
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

generate_video_pair("data/train/1/study/sax_10/")