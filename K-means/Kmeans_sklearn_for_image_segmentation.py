import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
import imutils

# read image as BGR (default)
im = cv2.imread(r"C:\Users\Acer\Desktop\CellTest\images\d1-b3-010.jpg")
# mask = cv2.imread(r"C:\Users\Acer\Desktop\CellTest\masks\d1-b3-010.png")

# convert BGR image to RGB
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
original_shape = im.shape
# print(im.shape)

# plt.imshow(im) # as RGB Format
# plt.show()

# flatten the image 2D for kmeans
# MxNx3 image to Kx3 image where K = MxN
all_pixels = im.reshape((-1, 3))
# print(all_pixels.shape)

########################################################### k-means

clusters = 3 # change accordingly to available distinct regions
column = 3  # column for plot # cant be lesser than 3 (or require extra editing)

# call kmeans library
# and fit the image into kmeans algo
km = KMeans(n_clusters=clusters)
km.fit(all_pixels)

# retrieve cluster centers
# in RGB format
centers = km.cluster_centers_
# print(centers)

# convert to integer format
centers = np.array(centers, dtype='uint8')

# print(centers)

########################################################### identify color of cluster cemters

i = 1

fig = plt.figure(1)

# Storing info in color array
colors = []
color_label = ['white', 'black', 'red', 'green', 'blue']

for each_col in centers:
    fig.add_subplot(2, column, i)
    plt.title(color_label[i - 1])
    plt.axis("off")
    i += 1

    colors.append(each_col)

    # color swatch
    a = np.zeros((100, 100, 3), dtype='uint8')
    a[:, :, :] = each_col
    plt.imshow(a)

# plt.show()

########################################################### plot new image according to cluster centers

# create empty image as same dimension of original image
new_img = np.zeros(all_pixels.shape, dtype='uint8')
# print(new_img.shape)

# color labels
label = km.labels_

# iterate over the image
# and plot based on color of cluster centers
# for ix in range(new_img.shape[0]):
#    new_img[ix] = colors[km.labels_[ix]]

# to distinguish clusters
for ix in range(new_img.shape[0]):
    if label[ix] == 0:
        new_img[ix] = [255, 255, 255]  # white
    elif label[ix] == 1:
        new_img[ix] = [0, 0, 0]  # black
    elif label[ix] == 2:
        new_img[ix] = [255, 0, 0]  # red
    elif label[ix] == 3:
        new_img[ix] = [0, 255, 0]  # green
    else:
        new_img[ix] = [0, 0, 255]  # blue

# reshape the new image to the original image dimensions (MxNx3)
new_img = new_img.reshape(original_shape)

########################################################### plot available images

# fig.add_subplot(2, column, column+1)
# plt.title('masks')
# plt.imshow(mask)
fig.add_subplot(2, column, column + 2)
plt.title('original')
plt.imshow(im)
fig.add_subplot(2, column, column + 3)
plt.title('segmented')
plt.imshow(new_img)

fig2 = plt.figure(2)
fig2.add_subplot(1, 2, 1)
plt.title('original')
plt.imshow(im)
fig2.add_subplot(1, 2, 2)
plt.title('segmented')
plt.imshow(new_img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
