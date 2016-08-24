import helper
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.measure import regionprops

_DEBUG = True

import random
random.seed(100)

def debug(msg, _type='info', tabs=1):
    end_ = '\033[0m'
    if not _DEBUG:
        return

    if tabs == 0:
        print '\033[1m' + msg + end_
        return

    if _type == 'info':
        res = '\033[94m' + msg + end_
    elif _type == 'warn':
        res = '\033[93m' + msg + end_
    else:
        res = msg

    print '\t' * tabs + res

def createImages(arr1, arr2, img1=None, img2=None, boxPosition=None):
    debug("Creating images", 'info', 1)
    warnings.filterwarnings("ignore", module="matplotlib")
    for i in xrange(len(arr1)):
        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(arr1[i], cmap=cm.Greys_r)
        if img2 is not None:
            helper.drawBox(boxPosition, ax)
        plt.axis('off')

        fig.add_subplot(2, 2, 2)
        plt.imshow(arr2[i], cmap=cm.Greys_r)
        plt.axis('off')

        if img1 is not None:
            fig.add_subplot(2, 2, 3)
            plt.imshow(img1, cmap=cm.Greys_r)
            plt.axis('off')

        if img2 is not None:
            fig.add_subplot(2, 2, 4)
            plt.imshow(img2, cmap=cm.Greys_r)
            plt.axis('off')

        fig.tight_layout()
        plt.savefig("videos_/tmp_file%02d.png" % i)

def removeCloseToBorders(location):
    copy_location = np.copy(location)
    step_y_u, step_y_d, step_x_r, step_x_l = 0.025, 0.05, 0.0125, 0.025
    p_u, p_d, p_r, p_l = 0, 0, 0, 0
    y, x = copy_location.shape
    step = 0
    previous_step = np.copy(copy_location)
    while step < 6:
        p_u, p_d, p_r, p_l, step = p_u + step_y_u, p_d + step_y_d, p_r + step_x_r, p_l + step_x_l, step + 1
        components, num_components = ndimage.label(copy_location)
        if num_components == 0:
            copy_location = previous_step

        previous_step = np.copy(copy_location)
        if num_components <= 1:
            continue

        y_min, y_max = p_u * y, (1 - p_d) * y
        x_min, x_max = p_r * x, (1 - p_l) * x

        copy_location[y_max:, :] = True
        copy_location[:y_min, :] = True
        copy_location[:, :x_min] = True
        copy_location[:, x_max:] = True

        clear_border(copy_location)

    debug('Shrinked %d times' % step, 'info', 2)
    img = ndimage.binary_fill_holes(copy_location)
    img = ndimage.binary_opening(img, structure=np.ones((6, 6)))
    return img

def findBoxPosition(mask):
    boxPosition = None
    num_regions = 0
    for region in regionprops(mask):
        if region.area < 100: continue

        num_regions += 1
        boxPosition = region.bbox

    if num_regions > 1:
        print 'Number of regions', num_regions
    return boxPosition

def findHeartPosition(images):
    debug("Locating heart position", 'info', 1)
    # the bigger the percentage - the less dots you have
    # try to find a good percentage
    percent, attempts = 0.04, 50
    have_found_heart = False
    for i in xrange(attempts):
        images_original, images_change, img_changes = helper.locateMovementFromImages(images, percent)

        crude_location = helper.removeSmallFeatures(img_changes, 0.005)
        crude_location = ndimage.binary_opening(crude_location, structure=np.ones((3, 3)))
        crude_location = helper.removeSmallFeatures(crude_location, 0.005)
        clean_location = removeCloseToBorders(crude_location)

        box_position = findBoxPosition(clean_location)
        if box_position is None:
            percent *= 0.75
            debug("Heart not found. Trying with another accuracy: %f" % percent, 'warn', 2)
        else:
            min_r, min_c, max_r, max_c = box_position
            rel_y, rel_x = (max_r - min_r) / float(crude_location.shape[0]), (max_c - min_c) / float(crude_location.shape[1])

            if max(rel_x, rel_y) > 0.6:
                alpha = 1 - (max(rel_x, rel_y) - 0.6) / 0.6
                percent /= alpha
                debug("Heart found. Region too big, adjusting accuracy: %f" % percent, 'warn', 2)
            elif min(rel_x, rel_y) < 0.14:
                alpha = 1 - (0.14 - min(rel_x, rel_y)) / 0.14
                percent *= alpha
                debug("Heart found. Region too small, adjusting accuracy: %f" % percent, 'warn', 2)
            else:
                debug("Heart found. Region looks good, moving on", 'info', 2)
                have_found_heart = True
                break


    if not have_found_heart:
        debug("Heart not found after %d attempts. Giving up" % attempts, 'warn', 2)
        return images_original, images_change, crude_location, clean_location, None



    components, num_components = ndimage.label(clean_location)
    sizes = ndimage.sum(clean_location, components, range(num_components + 1))
    maximum = max(sizes)
    mask_size = sizes < maximum / 5

    remove_pixel = mask_size[components]
    components[remove_pixel] = 0
    clean_location = components.astype(bool)
    box_position = findBoxPosition(clean_location)

    return images_original, images_change, crude_location, clean_location, box_position

def doAll():
    folders = helper.getRandomFolder(10)
    folders = helper.getFoldersOfFullExample(3)
    for folder in folders:
        debug("Analysing folder %s" % folder, 'info', 0)
        images = helper.readDICOM(folder)

        debug("Images read", 'info', 1)

        original, change, loc1, loc2, boxPosition = findHeartPosition(images)
        createImages(original, change, loc1, loc2, boxPosition)
        debug("Creating video", 'info', 1)
        helper. createVideoFromImages(folder.replace('/', '_')[:-1])
        debug("Removing images", 'info', 1)
        helper.deleteImages()

doAll()
