import os
import cv2

images = 'data/images/'
masks = 'data/masks/'
paths = (images, masks)
for path in paths:
    if os.path.exists(path):
        dirs = os.listdir(path)
        if len(dirs) != 0:
            for item in dirs:
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path):
                    select_img = full_path
                    file_name = os.path.basename(select_img)
                    chosen = file_name.split(".")
                    img = cv2.imread(full_path)
                    if path == images:
                        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab_img)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        cl_img = clahe.apply(l)
                        updated_lab_img2 = cv2.merge((cl_img, a, b))
                        cl_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
                        cv2.imwrite('preprocessed/images/' + chosen[0] + '.tiff', cl_img)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
                        cv2.imwrite('preprocessed/masks/' + chosen[0] + '.tiff', thresh)



