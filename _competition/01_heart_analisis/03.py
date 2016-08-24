import dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from skimage import feature
from scipy import ndimage

folder = "data/train/1/study/sax_10/"
images = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

fig = plt.figure()

img1 = images[0].pixel_array / 256.
img2 = images[1].pixel_array / 256.
img1_edges = feature.canny(img1, sigma=4)
img2_edges = feature.canny(img2, sigma=4)

fig.add_subplot(2, 4, 1)
plt.imshow(img1, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 2)
plt.imshow(img2, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 3)
diff = abs(img2 - img1) > 0.07
plt.imshow(diff, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 4)
plt.imshow(ndimage.binary_opening(diff), cmap=cm.Greys_r)

# =========== edges ===========
fig.add_subplot(2, 4, 5)
plt.imshow(img1_edges, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 6)
plt.imshow(img2_edges, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 7)
diff = abs(img2_edges - img1_edges) > 0.06
plt.imshow(diff, cmap=cm.Greys_r)

fig.add_subplot(2, 4, 8)
plt.imshow(ndimage.binary_closing(diff), cmap=cm.Greys_r)

plt.show()