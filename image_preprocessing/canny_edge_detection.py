import os 
import cv2 as cv
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(dir_path,"Sample images/Beautiful.png")

img=cv.imread(image_path,cv.IMREAD_GRAYSCALE)



img_edges = cv.Canny(img,1,1)

# plt.imshow(img_edges,cmap='gray')
# plt.show()

fig,axs=plt.subplots(1,2,figsize=(10, 5))
axs[0].imshow(img,cmap='gray')
axs[1].imshow(img_edges,cmap='gray')
plt.show()