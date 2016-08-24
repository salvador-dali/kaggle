import dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob

def generate_video(folder):
    img = [dicom.read_file(file_name) for file_name in glob.glob(folder + "*.dcm")]

    fig = plt.figure()

    fig.add_subplot(2, 2, 1)
    plt.imshow(img[0].pixel_array, cmap=cm.Greys_r)

    fig.add_subplot(2, 2, 2)
    plt.imshow(img[1].pixel_array, cmap=cm.Greys_r)

    fig.add_subplot(2, 2, 3)
    plt.imshow(img[2].pixel_array, cmap=cm.Greys_r)

    fig.add_subplot(2, 2, 4)
    plt.imshow(img[2].pixel_array, cmap=cm.Greys_r)

    plt.show()

    # for i in xrange(len(img)):
    #     plt.imshow(img[i].pixel_array, cmap=cm.Greys_r)
    #     plt.axis('off')
    #     plt.savefig(folder + "/file%02d.png" % i)

generate_video("data/train/1/study/sax_10/")