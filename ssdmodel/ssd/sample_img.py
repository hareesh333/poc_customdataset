import numpy as np
import argparse
import cv2
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Function to read custom annotations
def read_custom_annotations(file_path, img_width, img_height):
    annotations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().split()
            class_id = int(tokens[0])
            center_x = float(tokens[1]) * img_width
            center_y = float(tokens[2]) * img_height
            width = float(tokens[3]) * img_width
            height = float(tokens[4]) * img_height
            
            xmin = int(center_x - width / 2)
            ymin = int(center_y - height / 2)
            xmax = int(center_x + width / 2)
            ymax = int(center_y + height / 2)
            
            annotations.append((class_id, xmin, ymin, xmax, ymax))
    return annotations

# Function to calculate intersection over union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

# Function to match detections with ground truth
def match_detections_with_annotations(detections, annotations, iou_threshold=0.5):
    matches = []
    for detection in detections:
        for annotation in annotations:
            iou = calculate_iou(detection[1:], annotation[1:])
            if iou >= iou_threshold:
                matches.append((detection[0], annotation[0]))
                break
    return matches

# Construct the argument parser
parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')
parser.add_argument("--image", default="page_391-357124_14129_BFK_1_1E_2019_10.jpg", help="path to image file.")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt", help='Path to text network file.')
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel", help='Path to weights file.')
parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--annotations", default="page_391-357124_14129_BFK_1_1E_2019_10.txt", help="path to custom annotations file")
args = parser.parse_args()

# Labels of Network
classNames = {0: 'background', 1: 'Installation', 2: 'Sealant', 3: 'Press', 4: 'Cutter'}

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# Load image
frame = cv2.imread(args.image)
frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
heightFactor = frame.shape[0] / 300.0
widthFactor = frame.shape[1] / 300.0

# Create a blob from the image
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

# Set the blob as input to the network
net.setInput(blob)

# Perform detection
detections = net.forward()

# Parse custom annotations
annotations = read_custom_annotations(args.annotations, frame.shape[1], frame.shape[0])

# Store ground truth and predictions for evaluation
ground_truths = []
predictions = []

# Visualize SSD detections
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]  # Confidence of prediction
    if confidence > args.thr:  # Filter prediction
        class_id = int(detections[0, 0, i, 1])  # Class label

        # Object location
        x1 = int(detections[0, 0, i, 3] * 300)
        y1 = int(detections[0, 0, i, 4] * 300)
        x2 = int(detections[0, 0, i, 5] * 300)
        y2 = int(detections[0, 0, i, 6] * 300)

        x1 = int(widthFactor * x1)
        y1 = int(heightFactor * y1)
        x2 = int(widthFactor * x2)
        y2 = int(heightFactor * y2)

        predictions.append((class_id, x1, y1, x2, y2))

        # Draw location of object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label and confidence of prediction
        if class_id in classNames:
            label = f"{classNames[class_id]}: {confidence:.2f}"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, labelSize[1])
            cv2.rectangle(frame, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Visualize custom annotations
for annotation in annotations:
    class_id, xmin, ymin, xmax, ymax = annotation
    ground_truths.append((class_id, xmin, ymin, xmax, ymax))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    label = f"{classNames[class_id]}"
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ymin = max(ymin, labelSize[1])
    cv2.rectangle(frame, (xmin, ymin - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# Match detections with ground truth
matches = match_detections_with_annotations(predictions, ground_truths)

y_true = [match[1] for match in matches]
y_pred = [match[0] for match in matches]

# Calculate metrics
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Show the image with detections
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import numpy as np
# import argparse
# import cv2
# import os

# # Function to read custom annotations
# def read_custom_annotations(file_path, img_width, img_height):
#     annotations = []
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#         for line in lines:
#             tokens = line.strip().split()
#             class_id = int(tokens[0])
#             center_x = float(tokens[1]) * img_width
#             center_y = float(tokens[2]) * img_height
#             width = float(tokens[3]) * img_width
#             height = float(tokens[4]) * img_height
            
#             xmin = int(center_x - width / 2)
#             ymin = int(center_y - height / 2)
#             xmax = int(center_x + width / 2)
#             ymax = int(center_y + height / 2)
            
#             annotations.append((class_id, xmin, ymin, xmax, ymax))
#     return annotations

# # Construct the argument parser
# parser = argparse.ArgumentParser(description='Script to run MobileNet-SSD object detection network')
# parser.add_argument("--image", default="page_391-357124_14129_BFK_1_1E_2019_10.jpg", help="path to image file.")
# parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt", help='Path to text network file.')
# parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel", help='Path to weights file.')
# parser.add_argument("--thr", default=0.2, type=float, help="confidence threshold to filter out weak detections")
# parser.add_argument("--annotations", default="page_391-357124_14129_BFK_1_1E_2019_10.txt", help="path to custom annotations file")
# args = parser.parse_args()

# # Labels of Network
# classNames = {0: 'background', 1: 'Installation', 2: 'Sealant', 3: 'Press',4: 'Cutter'}

# # Load the Caffe model
# net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# # Load image
# frame = cv2.imread(args.image)
# frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
# heightFactor = frame.shape[0] / 300.0
# widthFactor = frame.shape[1] / 300.0

# # Create a blob from the image
# blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)

# # Set the blob as input to the network
# net.setInput(blob)

# # Perform detection
# detections = net.forward()

# # Parse custom annotations
# annotations = read_custom_annotations(args.annotations, frame.shape[1], frame.shape[0])

# # Visualize SSD detections
# for i in range(detections.shape[2]):
#     confidence = detections[0, 0, i, 2]  # Confidence of prediction
#     if confidence > args.thr:  # Filter prediction
#         class_id = int(detections[0, 0, i, 1])  # Class label

#         # Object location
#         x1 = int(detections[0, 0, i, 3] * 300)
#         y1 = int(detections[0, 0, i, 4] * 300)
#         x2 = int(detections[0, 0, i, 5] * 300)
#         y2 = int(detections[0, 0, i, 6] * 300)

#         x1 = int(widthFactor * x1)
#         y1 = int(heightFactor * y1)
#         x2 = int(widthFactor * x2)
#         y2 = int(heightFactor * y2)

#         # Draw location of object
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Draw label and confidence of prediction
#         if class_id in classNames:
#             label = f"{classNames[class_id]}: {confidence:.2f}"
#             labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             y1 = max(y1, labelSize[1])
#             cv2.rectangle(frame, (x1, y1 - labelSize[1]), (x1 + labelSize[0], y1 + baseLine), (255, 255, 255), cv2.FILLED)
#             cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# # Visualize custom annotations
# for annotation in annotations:
#     class_id, xmin, ymin, xmax, ymax = annotation
#     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
#     label = f"{classNames[class_id]}"
#     labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     ymin = max(ymin, labelSize[1])
#     cv2.rectangle(frame, (xmin, ymin - labelSize[1]), (xmin + labelSize[0], ymin + baseLine), (255, 255, 255), cv2.FILLED)
#     cv2.putText(frame, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#     print(label)
#     print(x1,y1,x2,y2)
# # Show the image with detections
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# cv2.imshow("frame", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



