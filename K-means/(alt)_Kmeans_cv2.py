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

# Number of clusters
k = 3

# Number of attempts, number of times algorithm is executed using different initial labelings.
# Algorithm return labels that yield best compactness.
# compactness : It is the sum of squared distance from each point to their corresponding centers.

attempts = 10

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
cv2.imwrite("k-means/segmented.jpg", res2)

numpy_horizontal_concat = np.concatenate((img, res2), axis=1)  # for image comparisons
numpy_horizontal_concat = imutils.resize(numpy_horizontal_concat, width=1500)

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (740, 500)
fontScale = 1
fontColor = (0, 0, 255)
lineType = 2

cv2.putText(numpy_horizontal_concat, str(k), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

cv2.imshow('kmeans', numpy_horizontal_concat)

cv2.waitKey(0)
cv2.destroyAllWindows()

# original & segmented
# plt.figure(figsize=(15, 15))
# plt.subplot(231)
# plt.title('Original Image')
# plt.imshow(img[:, :, 0], cmap='gray')
# plt.subplot(232)
# plt.title('Segmented Image')
# plt.imshow(res2[:, :, 0], cmap='gray')
#
# plt.show()

##################################################################################### canny edge detection

# img = cv2.imread(r"C:\Users\Acer\Desktop\CellTest\images\d1-b3-010.jpg", 0)

##################################################################################### auto
# auto canny (not much useful in some cases)
# v = np.median(img)
# sigma = 0.33
#
# #---- Apply automatic Canny edge detection using the computed median----
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))
# edges = cv2.Canny(img, lower, upper)

##################################################################################### trackbar
# canny trackbar (tested setup = (7, 22))
# img = imutils.resize(img, width=900) # resize the imported image
# edges = cv2.Canny(img,20,60)
#
# cv2.namedWindow('image')
# cv2.createTrackbar('Lower', 'image', 0, 255, nothing)  # for lower threshold trackbar
# cv2.createTrackbar('Upper', 'image', 0, 255, nothing)  # for upper threshold trackbar
#
# while 1:
#     numpy_horizontal_concat = np.concatenate((img, edges), axis=1)  # for image comparisons
#     cv2.imshow('image', numpy_horizontal_concat)
#
#     l = cv2.getTrackbarPos('Lower', 'image')
#     u = cv2.getTrackbarPos('Upper', 'image')
#
#     # adjust new canny value
#     edges = cv2.Canny(img, l, u)
#
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
#
# cv2.destroyAllWindows()

##################################################################################### compare canny
# original & canny
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2)
# plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
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
