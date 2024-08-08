# import numpy as np
# import cv2
# import torch
# import glob as glob

# from model import create_model

# # set the computation device
# device = torch.device('cpu')  # Use CPU only
# # load the model and the trained weights
# model = create_model(num_classes=5).to(device)
# model.load_state_dict(torch.load(
#     '/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/outputs/model250.pth', map_location=device
# ))
# model.eval()

# # directory where all the images are present
# DIR_TEST = '/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/test_data'
# test_images = glob.glob(f"{DIR_TEST}/*")
# print(f"Test instances: {len(test_images)}")

# # classes: 0 index is reserved for background
# CLASSES = [
#     'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
# ]

# # define the detection threshold...
# # ... any detection having score below this will be discarded
# detection_threshold = 0.8

# for i in range(len(test_images)):
#     # get the image file name for saving output later on
#     image_name = test_images[i].split('/')[-1].split('.')[0]
#     image = cv2.imread(test_images[i])
#     orig_image = image.copy()
#     # BGR to RGB
#     image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
#     # make the pixel range between 0 and 1
#     image /= 255.0
#     # bring color channels to front
#     image = np.transpose(image, (2, 0, 1)).astype(float)
#     # convert to tensor
#     image = torch.tensor(image, dtype=torch.float)
#     # add batch dimension
#     image = torch.unsqueeze(image, 0)
#     with torch.no_grad():
#         outputs = model(image)
    
#     # load all detection to CPU for further operations
#     outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
#     # carry further only if there are detected boxes
#     if len(outputs[0]['boxes']) != 0:
#         boxes = outputs[0]['boxes'].data.numpy()
#         scores = outputs[0]['scores'].data.numpy()
#         # filter out boxes according to `detection_threshold`
#         boxes = boxes[scores >= detection_threshold].astype(np.int32)
#         draw_boxes = boxes.copy()
#         # get all the predicted class names
#         pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
#         # draw the bounding boxes and write the class name on top of it
#         for j, box in enumerate(draw_boxes):
#             cv2.rectangle(orig_image,
#                         (int(box[0]), int(box[1])),
#                         (int(box[2]), int(box[3])),
#                         (0, 0, 255), 2)
#             cv2.putText(orig_image, pred_classes[j], 
#                         (int(box[0]), int(box[1]-5)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
#                         2, lineType=cv2.LINE_AA)

#         cv2.imshow('Prediction', orig_image)
#         cv2.waitKey(1)
#         cv2.imwrite(f"/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/test_predictions/{image_name}.jpg", orig_image, )
#     print(f"Image {i+1} done...")
#     print('-'*50)

# print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()
import torch
import cv2
import numpy as np
import glob
from model import create_model
from config import DEVICE, NUM_CLASSES
import torchvision.transforms as transforms

# Set the computation device
device = torch.device(DEVICE)

# Load the model and the trained weights
model = create_model(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load('/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/outputs/model250.pth', map_location=device))
model.eval()

# Directory where all the images are present
DIR_TEST = '/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")

# Classes: 0 index is reserved for background
CLASSES = [
    'background', 'Arduino_Nano', 'ESP8266', 'Raspberry_Pi_3', 'Heltec_ESP32_Lora'
]

# Define the detection threshold
detection_threshold = 0.8

# Create a transformation pipeline for the input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

for i in range(len(test_images)):
    # Get the image file name for saving output later on
    image_name = test_images[i].split('/')[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    
    # Apply the transformation
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    
    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
    
    # Load all detection to CPU for further operations
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicted class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j], 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                        2, lineType=cv2.LINE_AA)

        
        # Save the output image
        cv2.imwrite(f"/home/harish/Documents/Custom_dataset_Practice/pytorchdemo/customdatast/test_predictions/{image_name}.jpg", orig_image)
    
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')