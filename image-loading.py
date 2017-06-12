from __future__ import division, print_function

import numpy
import pylab
from PIL import Image

img = Image.open(open("/home/faruk/Desktop/wang1000/0/0.jpg"))
img = numpy.asarray(img, dtype='float64') / 256.

print(img)

# put image in 4D tensor of shape (1, 3, height, width)
img_tensor = img.transpose(2, 0, 1).reshape(1, 3, 384, 256)
filtered_img = img_tensor

# plot original image and first and second components of output
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()