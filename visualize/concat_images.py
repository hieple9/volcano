from matplotlib import pyplot
from scipy.misc import toimage
import numpy as np
from scipy import misc

name = "images/layer_conv1d_8_filter%s.png"
all_images = []
for i in range(127):
    image = misc.imread(name % i)
    all_images.append(image)
all_images = np.array(all_images)

# Create a grid of 11x11 images
for i in range(121):
    pyplot.subplot(11, 11, i+1)
    pyplot.imshow(toimage(all_images[i]))
pyplot.show()
