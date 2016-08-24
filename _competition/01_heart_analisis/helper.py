import dicom
import glob
import os
import subprocess
import random
import numpy as np
from scipy import ndimage
import matplotlib.patches as mpatches


def readDICOM(folder):
    images = [dicom.read_file(file_name).pixel_array / 256. for file_name in glob.glob(folder + "*.dcm")]
    x, y = images[0].shape
    if x < y: images = [np.rot90(i) for i in images]

    return images

def getRandomFolder(num=1):
    folders = []
    for i in glob.glob('data/train/*/'):
        for location in glob.glob(i + 'study/sax_*'):
            folders.append(location + '/')

    return random.sample(folders, num)

def getFoldersOfFullExample(num):
    folders = []
    for i in glob.glob('data/train/' + str(num) + '/'):
        for location in glob.glob(i + 'study/sax_*'):
            folders.append(location + '/')
    return folders

def createVideoFromImages(file_name='video'):
    file_name += '.mp4'
    os.chdir('videos_')
    if os.path.exists(file_name):
        os.remove(file_name)

    command = 'ffmpeg -framerate 5 -i tmp_file%02d.png -r 30 -pix_fmt yuv420p ' + file_name
    subprocess.call(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chdir('..')

def deleteImages():
    os.chdir('videos_')
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    os.chdir('..')

def locateMovementFromImages(images, threshold):
    """Returns
    - array of original images
    - array of movements
    - compound movement
    """
    original, compounds, movement_map = [], [], np.zeros(images[0].shape, dtype=bool)
    for i in xrange(len(images) - 1):
        img_prev, img_next = images[i], images[i + 1]
        movement_map |= ndimage.binary_erosion(abs(img_next - img_prev) > threshold)
        original.append(img_prev)
        compounds.append(movement_map.copy())

    return original, compounds, movement_map

def drawBox(boxPosition, ax):
    if boxPosition:
        min_r, min_c, max_r, max_c = boxPosition
        rect = mpatches.Rectangle(
            (min_c, min_r), max_c - min_c, max_r - min_r,
            fill=False,
            edgecolor='blue',
            linewidth=3
        )
        ax.add_patch(rect)

def removeSmallFeatures(compound, percents):
    maximum = compound.shape[0] * compound.shape[1] * percents

    components, num_components = ndimage.label(compound)
    sizes = ndimage.sum(compound, components, range(num_components + 1))
    mask_size = sizes < maximum

    remove_pixel = mask_size[components]
    components[remove_pixel] = 0
    return components.astype(bool)
