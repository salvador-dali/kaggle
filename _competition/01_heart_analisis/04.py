import dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from scipy import ndimage

folder = "data/train/1/study/sax_10/"
images = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

fig = plt.figure()

l, threshold = 8, 0.07
for i in xrange(l):
    img_prev = images[i].pixel_array / 256.
    img_next = images[i + 1].pixel_array / 256.

    diff = abs(img_next - img_prev) > threshold
    fig.add_subplot(2, l, i + 1)
    plt.imshow(diff, cmap=cm.Greys_r)

    fig.add_subplot(2, l, l + i + 1)
    plt.imshow(ndimage.binary_opening(diff), cmap=cm.Greys_r)

plt.show()