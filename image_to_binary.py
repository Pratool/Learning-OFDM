import cv2
import numpy as np

im_gray = cv2.imread('cats.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)

(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

binary_list = []
new_binary_list = []
# print len(im_bw)
print len(im_bw)

for i in range(len(im_bw)):
    for j in range(len(im_bw[i])):
        if (im_bw[i][j] == 0):
            binary_list.append(0)
        else:
            binary_list.append(1)

for n in range(len(binary_list)):
    if (n % 14 == 0):
        new_binary_list.append(1)
        new_binary_list.append(1)
        new_binary_list.append(binary_list[n])
    else:
        new_binary_list.append(binary_list[n])

print new_binary_list
print len(new_binary_list)

#GETTING THE IMAGE BACK:
initial_image_list = []
image_list = []
for m in range(len(new_binary_list)):
    if (m %16 != 0 or m %17 != 0):
        initial_image_list.append(new_binary_list[m])        

for k in range(len(im_bw)):
    temp_list = []
    for n in range(k*(len(initial_image_list)/len(im_bw)), k*(len(initial_image_list)/len(im_bw)) + (len(initial_image_list)/len(im_bw))):
        if (initial_image_list[n] == 0):
            temp_list.append(0)
        else:
            temp_list.append (255)
    image_list.append(temp_list)

cv2.imwrite('bw_image.png', np.array(image_list))