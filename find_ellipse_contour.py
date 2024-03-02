import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./shapes_dataset_HR/1/test0037.png', cv2.IMREAD_GRAYSCALE)  # Convert image to grayscale
org_img = img.copy()
plt.imshow(org_img, cmap='gray')

kernel_square = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_square, iterations=1)

contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)

largest_ellipse = None
max_area = 0

for contour in contours:
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        area = ellipse[1][0] * ellipse[1][1] * np.pi  # Calculate the area of the fitted ellipse
        if area > max_area:
            max_area = area
            largest_ellipse = ellipse

if largest_ellipse is not None:
    cv2.ellipse(img, largest_ellipse, (0, 255, 0), 2)
else:
    print('No ellipses found')

plt.show()
plt.imshow(img)
