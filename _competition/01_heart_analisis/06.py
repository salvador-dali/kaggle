import dicom, glob, os, subprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import warnings
import random
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import matplotlib.patches as mpatches

def prepare_imgs(folder):
    images = [dicom.read_file(file_name).pixel_array / 256. for file_name in glob.glob(folder + "*.dcm")]

    # rotate if needed
    x, y = images[0].shape
    if x < y: images = [np.rot90(i) for i in images]

    return images

def generate_slices(images, threshold):
    original, compounds, compound = [], [], np.zeros(images[0].shape, dtype=bool)
    for i in xrange(len(images) - 1):
        img_prev, img_next = images[i], images[i + 1]
        compound |= ndimage.binary_erosion(abs(img_next - img_prev) > threshold)
        original.append(img_prev)
        compounds.append(compound.copy())

    return original, compounds, compound

def locateHeartPosition(compound):
    maximum = compound.shape[0] * compound.shape[1] / 200

    components, num_components = ndimage.label(compound)
    sizes = ndimage.sum(compound, components, range(num_components + 1))
    mask_size = sizes < maximum

    # print "Components", sorted([int(i) for i in list(sizes)], reverse=True)
    remove_pixel = mask_size[components]
    components[remove_pixel] = 0
    components = components.astype(bool)

    location = ndimage.binary_dilation(components, structure=np.ones((3, 3)))
    return ndimage.binary_erosion(location)

def removeUpDown(location):
    _, num_components = ndimage.label(location)
    if num_components == 1:
        return ndimage.binary_fill_holes(location)

    y, x = location.shape
    margin_x = 0.1
    margin_y_up, margin_y_down = 0.15, 0.25
    y_min, y_max = margin_y_up * y, (1 - margin_y_down) * y
    x_min, x_max = margin_x * x, (1 - margin_x) * x

    cleaned = np.copy(location)
    cleaned[y_max:,:] = True
    cleaned[:y_min,:] = True

    cleaned[:,:x_min] = True
    cleaned[:,x_max:] = True

    clear_border(cleaned)
    return ndimage.binary_fill_holes(cleaned)

def drawbox(mask, ax):
    for region in regionprops(mask):
        if region.area < 100:
            continue

        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle(
            (minc, minr), maxc - minc, maxr - minr,
            fill=False, edgecolor='red', linewidth=3
        )
        ax.add_patch(rect)

def generate_imgs(folder, threshold=0.04):
    images_original, images_compound, compound = generate_slices(prepare_imgs(folder), threshold)

    location = locateHeartPosition(compound)
    clean_location = removeUpDown(location)

    warnings.filterwarnings("ignore", module="matplotlib")
    for i in xrange(len(images_original)):
        fig = plt.figure()
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(images_original[i], cmap=cm.Greys_r)
        drawbox(clean_location, ax)
        plt.axis('off')

        fig.add_subplot(2, 2, 2)
        plt.imshow(images_compound[i], cmap=cm.Greys_r)
        plt.axis('off')

        # =====================
        fig.add_subplot(2, 2, 3)
        plt.imshow(location, cmap=cm.Greys_r)

        fig.add_subplot(2, 2, 4)
        plt.imshow(clean_location, cmap=cm.Greys_r)

        # =====================
        fig.tight_layout()
        # plt.show()
        # return
        plt.savefig("videos_/tmp_file%02d.png" % i)

def generate_video():
    os.chdir('videos_')
    file_name = "_video_.mp4"
    if os.path.exists(file_name):
        os.remove(file_name)

    command = 'ffmpeg -framerate 5 -i tmp_file%02d.png -r 30 -pix_fmt yuv420p ' + file_name
    subprocess.call(command.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for file_name in glob.glob("*.png"):
        os.remove(file_name)

    os.chdir('..')

def getRandomFolder():
    folders = []
    for i in glob.glob('data/train/*/'):
        for location in glob.glob(i + 'study/sax_*'):
            folders.append(location + '/')

    return random.choice(folders)



folder = getRandomFolder()
print folder
generate_imgs(folder)
generate_video()