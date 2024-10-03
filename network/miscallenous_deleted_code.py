import torchvision.transforms as transforms
import matplotlib.pyplot as plt



to_pil = transforms.ToPILImage()
def display_tensor_as_img(img):
    img=to_pil(img)
    plt.imshow(img, cmap='gray')
    plt.show()




if(-15.0<=angle<=15.0):
        angle=0.0



# transforms.Lambda(lambda img: 1-img), # Invert image to black with white background

PAD_FILL_VALUE=1 # Set the pad fill value to white






import matplotlib.pyplot as plt
import torchvision.transforms as transforms
to_pil = transforms.ToPILImage()


# for imgs, labels in train_dataloader:
#     print(imgs.shape)
#     imgs=imgs[0]
#     img=to_pil(imgs)
#     plt.imshow(img, cmap='gray')
#     plt.show()

    
#     exit()
#     print(imgs.shape)
#     exit()
