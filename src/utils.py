import cv2
import numpy as np
from collections import OrderedDict

CLASS_IDS = OrderedDict()
CLASS_IDS[8] = "apple"
CLASS_IDS[35] = "book"
CLASS_IDS[38] = "bowtie"
CLASS_IDS[58] = "candle"
CLASS_IDS[74] = "cloud"
CLASS_IDS[87] = "cup"
CLASS_IDS[94] = "door"
CLASS_IDS[104] = "envelope"
CLASS_IDS[107] = "eyeglasses"
CLASS_IDS[136] = "hammer"
CLASS_IDS[139] = "hat"
CLASS_IDS[156] = "ice cream"
CLASS_IDS[167] = "leaf"
CLASS_IDS[252] = "scissors"
CLASS_IDS[283] = "star"
CLASS_IDS[301] = "t-shirt"
CLASS_IDS[209] = "pants"
CLASS_IDS[323] = "tree"

def get_images(path, classes):
    images = [cv2.imread("{}/{}.png".format(path, item), cv2.IMREAD_UNCHANGED) for item in classes]
    return images
    # Existing get_images function...

def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    fg_image = cv2.resize(fg_image, sizes)
    fg_mask = fg_image[:, :, 3:]
    fg_image = fg_image[:, :, :3]
    bg_mask = 255 - fg_mask
    bg_image = bg_image / 255
    fg_image = fg_image / 255
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255
    image = cv2.addWeighted(bg_image * bg_mask, 255, fg_image * fg_mask, 255, 0.).astype(np.uint8)
    return image
    # Existing get_overlay function...

# def preprocess_image(image, size=(28, 28)):
#     """
#     Preprocess the image for model prediction.
#     Args:
#     - image: The image to preprocess.
#     - size: The target size of the image.
    
#     Returns:
#     - The preprocessed image.
#     """
#     # Resize the image
#     image = cv2.resize(image, size)
    
#     # Convert to grayscale if your model expects grayscale images
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Normalize pixel values to be between 0 and 1
#     image = image.astype(np.float32) / 255.0
    
#     # Add a channel dimension if the model expects it
#     if len(image.shape) == 2:
#         image = np.expand_dims(image, axis=-1)
    
#     return image