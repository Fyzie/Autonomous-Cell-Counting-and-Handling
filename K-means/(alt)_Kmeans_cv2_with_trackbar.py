import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt

def nothing(x):
    pass

img = cv2.imread(r"C:\Users\Acer\Desktop\CellTest\images\d1-b3-010.jpg")
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1, 3))  # -1 reshape means, in this case MxN

# We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
img2 = np.float32(img2)

# Define criteria, number of clusters and apply k-means
# When this criterion is satisfied, the algorithm iteration stops.
# cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
# Max iterations, in this example 10.
# Epsilon, required accuracy, in this example 1.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
res2 = img

# Number of clusters
k = 1

# Number of attempts, number of times algorithm is executed using different initial labelings.
# Algorithm return labels that yield best compactness.
# compactness : It is the sum of squared distance from each point to their corresponding centers.

attempts = 10

cv2.namedWindow('kmeans')
cv2.createTrackbar('Cluster', 'kmeans', 2, 10, nothing)  # for clustertrackbar
cv2.createTrackbar('Lower', 'kmeans', 0, 255, nothing)  # for lower threshold trackbar
cv2.createTrackbar('Higher', 'kmeans', 0, 255, nothing)  # for higher threshold trackbar

# res2 = cv2.Canny(img, 7, 22)

while 1:
    numpy_horizontal_concat = np.concatenate((img, res2), axis=1)  # for image comparisons
    numpy_horizontal_concat = imutils.resize(numpy_horizontal_concat, width=1500)

    cv2.imshow('kmeans', numpy_horizontal_concat)

    k = cv2.getTrackbarPos('Cluster', 'kmeans')
    l = cv2.getTrackbarPos('Lower', 'kmeans')
    h = cv2.getTrackbarPos('Higher', 'kmeans')

    # other flags needed as inputs for K-means
    # Specify how initial seeds are taken.
    # Two options, cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS

    ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # cv2.kmeans outputs 2 parameters.
    # 1 Compactness.
    # 2 Labels: Label array.
    # 3 Center. the array of centers of clusters. For k=4 we will have 4 centers.
    # For RGB image, we will have center for each image, so tota 4x3 = 12.
    # Now convert center values from float32 back into uint8.
    center = np.uint8(center)

    # Next, we have to access the labels to regenerate the clustered image
    res = center[label.flatten()]

    res2 = res.reshape(img.shape)  # Reshape labels to the size of original image
    # res2 = cv2.Canny(res2, 7, 22)
    rgb = cv2.cvtColor(res2, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # _, binary = cv2.threshold(gray, l, h, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res2 = cv2.drawContours(res2, contours, -1, (0, 255, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cv2.imwrite("k-means/segmented.jpg", res2)

#original & segmented
# plt.figure(figsize=(15, 15))
# plt.subplot(231)
# plt.title('Original Image')
# plt.imshow(img[:, :, 0], cmap='gray')
# plt.subplot(232)
# plt.title('Segmented Image')
# plt.imshow(res2[:, :, 0], cmap='gray')
#
# plt.show()

##################################################################################### disable cluster

# # disable only the cluster number 2 (turn the pixel into black)
# masked_image = np.copy(res2)
# # convert to the shape of a vector of pixel values
# masked_image = masked_image.reshape((-1, 3))
# # color (i.e cluster) to disable
# cluster = 2
# masked_image[label == cluster] = [0, 0, 0]
# # convert back to original shape
# masked_image = masked_image.reshape(res2.shape)
# # show the image
# plt.imshow(masked_image)
# plt.show()
