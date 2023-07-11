
import numpy as np
import imutils
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours

img_path = "images/1.jpg"

# Read image and preprocess
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image in grayscale
blur = cv2.GaussianBlur(gray, (9, 9), 0)

# it will detect t the edge
edged = cv2.Canny(blur, 50, 100)  # cv2.Canny(img, threshold_lower, threshold_upper)

# Because, erosion removes white noises, but it also shrinks our object.
# So we dilate it. Since noise is gone, they won't come back,
# but our object area increases. It is also useful in joining broken parts of an object.
kernel = np.ones((5, 5), np.uint8)
edged = cv2.dilate(edged, kernel, iterations=1) #img_erosion = cv2.erode(img, kernel, iterations=1)
edged = cv2.erode(edged, kernel, iterations=1)

# Find contours
# Contours are defined as the line joining all the points along the boundary of an image that are having the same intensity

# cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
# this returns img,contours,hierarchy
# finding contours is like finding white object from black background.
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# function of imutils.grab_contours, returning counters (contours) in cnts
# opencv2 returns two values: contours and hierarchy. opencv3 returns three values: img (image), countours (outline), hierarchy (hierarchy).
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts,temp) = contours.sort_contours(cnts)

# # Remove contours which are small
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

#cv2.drawContours(image, cnts, -1, (0,255,0), 3)

#show_images([image, edged])
#print(len(cnts))

# Reference object dimensions
# # Here for reference I have used a 2cm x 2cm square
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object) #The function minAreaRect() calculates the minimum area rectangle that encloses the given points or contour.
# Now Box is my refrence object
box = cv2.boxPoints(box) # it will take only border points

# print(type(box))  # type = numpy.ndarray
# print(box)

# it is used for ordering co_ordinates clock_wise
box = perspective.order_points(box)

(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2.3
pixel_per_cm = dist_in_pixel/dist_in_cm

# Draw remaining contours
for cnt in cnts:
	# drawing borders
	box = cv2.minAreaRect(cnt)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	# print(box) = [[168. 123.]
	#              [287. 123.]
	#              [287. 241.]
	#              [168. 241.]]
	#DrawContours(img, contour, external_color, hole_color, max_level)
	cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 3)


	mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
	mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
	wid = euclidean(tl, tr)/pixel_per_cm
	ht = euclidean(tr, br)/pixel_per_cm

	# to print the length as text
	text_wid = "{w:.2f}cm".format(w=wid)
	text_ht = "{h:.2f}cm".format(h=ht)
	# cv2.putText(img, text, co_ordinates, fontFace, fontScale, color, thickness)
	cv2.putText(image, text_wid, (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0 ,0), 2)
	cv2.putText(image, text_ht, (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


#show_images([image])
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("output", 800, 600)
cv2.imshow("output", image)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()


"""
Usage: This script will measure different objects in the frame using a reference object of known dimension.
The object with known dimension must be the leftmost object.
"""