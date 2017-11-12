from model import *

img_path = 'data/Data_Collection/SCUT-FBP-110.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
plot.imshow(img)
plot.show()