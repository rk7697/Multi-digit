# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt



# to_pil = transforms.ToPILImage()
# def display_tensor_as_img(img):
#     img=to_pil(img)
#     plt.imshow(img, cmap='gray')
#     plt.show()




# if(-15.0<=angle<=15.0):
#         angle=0.0



# # transforms.Lambda(lambda img: 1-img), # Invert image to black with white background

# PAD_FILL_VALUE=1 # Set the pad fill value to white






# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
# to_pil = transforms.ToPILImage()


# # for imgs, labels in train_dataloader:
# #     print(imgs.shape)
# #     imgs=imgs[0]
# #     img=to_pil(imgs)
# #     plt.imshow(img, cmap='gray')
# #     plt.show()

    
# #     exit()
# #     print(imgs.shape)
# #     exit()


# # transforms.Lambda(lambda img: canny_edge_detection(img)),

# # Apply canny edge detection to image
# def canny_edge_detection(image):
#     img_array = np.array(image) # Convert PIL image to numpy

#     # Set threshold for weak and strong edges to 1
#     threshold_1 = 1
#     threshold_2 = 10
#     img_edges = Canny(img_array, threshold1=threshold_1, threshold2=threshold_2)
#     return img_edges

# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# to_pil = transforms.ToPILImage()
# def display_tensor_as_img(img):
#     img=to_pil(img)
#     plt.imshow(img, cmap='gray')
#     plt.show()


# def display_tensor_as_img(img, box):
#     image=to_pil(img)
#     center_x, center_y, width, height = box

#     left = center_x - width / 2
#     top = center_y - height / 2
#     fig, ax = plt.subplots()

#     #    Display the image
#     ax.imshow(image, cmap = 'gray')

#     # Create a rectangle representing the bounding box
#     rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='red', facecolor='none')

#     # Add the rectangle to the plot
#     ax.add_patch(rect)

#     # Display the result
#     plt.axis('off')
#     plt.show()

# print(imgs)
#             print(bboxes)
#             print(labels)
#             image=imgs[0]
#             box = np.array(bboxes[0])
#             display_tensor_as_img(image, box)

