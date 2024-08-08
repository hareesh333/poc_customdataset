# from sklearn.metrics import precision_score, recall_score
# from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
# from config import VISUALIZE_TRANSFORMED_IMAGES
# from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
# from model import create_model
# from utils import Averager
# from tqdm.auto import tqdm
# from datasets import train_loader, valid_loader

# import torch
# import matplotlib.pyplot as plt
# import time
# from torchvision.ops import box_iou
# plt.style.use('ggplot')

# def calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
#     # Sort predictions by scores in descending order
#     sorted_indices = torch.argsort(pred_scores, descending=True)
#     pred_boxes = pred_boxes[sorted_indices]
#     pred_labels = pred_labels[sorted_indices]
#     pred_scores = pred_scores[sorted_indices]

#     # Calculate IoU between predictions and ground truths
#     ious = box_iou(pred_boxes, gt_boxes)

#     # Determine matches based on IoU and class label
#     true_positives = 0
#     gt_matched = set()
#     for i, iou in enumerate(ious):
#         max_iou, max_index = torch.max(iou, 0)
#         if max_iou > iou_threshold and pred_labels[i] == gt_labels[max_index]:
#             if max_index.item() not in gt_matched:
#                 true_positives += 1
#                 gt_matched.add(max_index.item())

#     precision = true_positives / len(pred_labels) if len(pred_labels) > 0 else 0
#     recall = true_positives / len(gt_labels) if len(gt_labels) > 0 else 0
#     return precision, recall


# # Updated training function with precision and recall
# def train(train_data_loader, model):
#     print('Training')
#     model.train()  # Make sure the model is in training mode
#     global train_itr
#     global train_loss_list
#     global train_precision_list
#     global train_recall_list
    
#     # Initialize tqdm progress bar
#     prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
#     all_precisions = []
#     all_recalls = []

#     for i, data in enumerate(prog_bar):
#         optimizer.zero_grad()
#         images, targets = data
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         # Obtain the loss and model predictions
#         loss_dict = model(images, targets)  # This assumes the model returns a dict with loss components
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#         train_loss_list.append(loss_value)

#         # Backpropagate the errors to update model weights
#         losses.backward()
#         optimizer.step()

#         # Temporarily switch to evaluation mode for calculating precision and recall
#         model.eval()
#         with torch.no_grad():
#             outputs = model(images)
#             for output, target in zip(outputs, targets):
#                 pred_boxes = output['boxes']
#                 pred_labels = output['labels']
#                 pred_scores = output['scores']

#                 gt_boxes = target['boxes']
#                 gt_labels = target['labels']

#                 precision, recall = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
#                 all_precisions.append(precision)
#                 all_recalls.append(recall)

#         # Switch back to training mode
#         model.train()

#         mean_precision = torch.tensor(all_precisions, dtype=torch.float).mean().item()
#         mean_recall = torch.tensor(all_recalls, dtype=torch.float).mean().item()
#         train_loss_hist.send(loss_value)
#         train_precision_hist.send(mean_precision)
#         train_recall_hist.send(mean_recall)
        
#         # Update progress bar description with loss and metric
#         prog_bar.set_description(desc=f"Loss: {loss_value:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")

#         train_itr += 1

#     train_precision_list.extend(all_precisions)
#     train_recall_list.extend(all_recalls)

#     return train_loss_list, train_precision_list, train_recall_list

# def validate(valid_data_loader, model):
#     print('Validating')
#     global val_itr
#     global val_loss_list
#     global val_precision_list
#     global val_recall_list
    
    
#     # Initialize tqdm progress bar
#     prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

#     all_precisions = []
#     all_recalls = []
    
#     for i, data in enumerate(prog_bar):
#         images, targets = data
        
#         # Switch back to training mode
#         model.train()
        
#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#         loss_dict = model(images, targets)
#         print(loss_dict, type(loss_dict))
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#         val_loss_list.append(loss_value)
        
        
#         with torch.no_grad():
#             model.eval()
#             outputs = model(images)
        

#         for output, target in zip(outputs, targets):
#             pred_boxes = output['boxes']
#             pred_labels = output['labels']
#             pred_scores = output['scores']
            
#             gt_boxes = target['boxes']
#             gt_labels = target['labels']
            
#             precision, recall = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
#             all_precisions.append(precision)
#             all_recalls.append(recall)
        
#         # Update progress bar
#         mean_precision = torch.tensor(all_precisions, dtype=torch.float).mean().item()
#         mean_recall = torch.tensor(all_recalls, dtype=torch.float).mean().item()

#         val_loss_hist.send(loss_value)
#         val_precision_hist.send(mean_precision)
#         val_recall_hist.send(mean_recall)
        
#         prog_bar.set_description(desc=f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")
#         val_itr += 1
        
#     val_precision_list.extend(all_precisions)
#     val_recall_list.extend(all_recalls)

#     return val_loss_list, val_precision_list, val_recall_list


# if __name__ == '__main__':
#     # initialize the model and move to the computation device
#     model = create_model(num_classes=NUM_CLASSES)
#     model = model.to(DEVICE)
    
#     # get the model parameters
#     params = [p for p in model.parameters() if p.requires_grad]
    
#     # define the optimizer
#     optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
#     # initialize the Averager class
#     train_loss_hist = Averager()
#     train_precision_hist = Averager()
#     train_recall_hist = Averager()
#     val_loss_hist = Averager()
#     val_precision_hist = Averager()
#     val_recall_hist = Averager()
#     train_itr = 1
#     val_itr = 1
    
    
#     # train and validation loss lists to store loss values of all...
#     # ... iterations till end and plot graphs for all iterations
#     train_loss_list = []
#     train_precision_list = []
#     train_recall_list = []
#     val_loss_list = []
#     val_precision_list = []
#     val_recall_list = []
#     # name to save the trained model with
#     MODEL_NAME = 'model'
#     # whether to show transformed images from data loader or not
#     if VISUALIZE_TRANSFORMED_IMAGES:
#         from utils import show_tranformed_image
#         show_tranformed_image(train_loader)
#     # start the training epochs
#     for epoch in range(NUM_EPOCHS):
#         print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
#         # reset the training and validation loss histories for the current epoch
#         train_loss_hist.reset()
#         val_loss_hist.reset()
#         # create two subplots, one for each, training and validation
#         figure_1, train_ax = plt.subplots()
#         figure_2, valid_ax = plt.subplots()
#         # start timer and carry out training and validation
#         start = time.time()
#         train_loss, train_precision, train_recall = train(train_loader, model)
        
#         val_loss, val_precision, val_recall = validate(valid_loader, model)
#         # train loss and metrics
#         print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
#         print(f"Epoch #{epoch} train precision: {train_precision_hist.value:.3f}")
#         print(f"Epoch #{epoch} train recall: {train_recall_hist.value:.3f}")

#         # validation loss and metrics        
#         print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
#         print(f"Epoch #{epoch} validation precision: {val_precision_hist.value:.4f}")
#         print(f"Epoch #{epoch} validation recall: {val_recall_hist.value:.4f}")
#         end = time.time()
#         print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
#         if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
#             torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
#             print('SAVING MODEL COMPLETE...\n')
        
#         if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
#             train_ax.plot(train_loss, color='blue')
#             train_ax.set_xlabel('iterations')
#             train_ax.set_ylabel('train loss')
#             valid_ax.plot(val_loss, color='red')
#             valid_ax.set_xlabel('iterations')
#             valid_ax.set_ylabel('validation loss')
#             figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
#             figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
#             print('SAVING PLOTS COMPLETE...')
        
#         if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
#             train_ax.plot(train_loss, color='blue')
#             train_ax.set_xlabel('iterations')
#             train_ax.set_ylabel('train loss')
#             valid_ax.plot(val_loss, color='red')
#             valid_ax.set_xlabel('iterations')
#             valid_ax.set_ylabel('validation loss')
#             figure_1.savefig(f"{OUT_DIR}/train_loss_{epoch+1}.png")
#             figure_2.savefig(f"{OUT_DIR}/valid_loss_{epoch+1}.png")
#             torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        
#         plt.close('all')
# Import precision_score and recall_score from sklearn.metrics for calculating precision and recall
from sklearn.metrics import precision_score, recall_score

# Import various configurations and constants from config.py
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES, SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH, EARLY_STOPPING, PATIENCE,EARLY_STOPPING_METRIC,GRADIENT_ACCUMULATION_STEPS

# Import the function to create a model from model.py
from model import create_model

# Import Averager class from utils.py for tracking average values
from utils import Averager

# Import tqdm for progress bar during training and validation
from tqdm.auto import tqdm

# Import data loaders from datasets.py
from datasets import train_loader, valid_loader

# Import PyTorch library for deep learning operations
import torch

# Import amp module for mixed precision training from PyTorch
import torch.cuda.amp as amp

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import time for tracking duration of operations
import time

# Import box_iou from torchvision for calculating intersection over union of bounding boxes
from torchvision.ops import box_iou

# Set the plotting style to 'ggplot'
plt.style.use('ggplot')

# Function to calculate precision and recall based on predicted and ground truth bounding boxes
def calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    # Sort predictions by scores in descending order
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    # Compute IOU between predicted and ground truth boxes
    ious = box_iou(pred_boxes, gt_boxes)

    true_positives = 0
    gt_matched = set()
    
    # Iterate through each prediction
    for i, iou in enumerate(ious):
        # Find the maximum IOU and its corresponding ground truth index
        max_iou, max_index = torch.max(iou, 0)
        
        # Check if IOU exceeds threshold and labels match
        if max_iou > iou_threshold and pred_labels[i] == gt_labels[max_index]:
            # If the ground truth box hasn't been matched yet, count as true positive
            if max_index.item() not in gt_matched:
                true_positives += 1
                gt_matched.add(max_index.item())

    # Calculate precision and recall
    precision = true_positives / len(pred_labels) if len(pred_labels) > 0 else 0
    recall = true_positives / len(gt_labels) if len(gt_labels) > 0 else 0
    return precision, recall

# Training function
def train(train_data_loader, model, optimizer, scaler):
    print('Training')  # Print start of training
    model.train()  # Set model to training mode
    global train_itr  # Use global variable for iteration tracking
    global train_loss_list
    global train_precision_list
    global train_recall_list
    
    # Create a progress bar for training
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    all_precisions = []
    all_recalls = []
    accumulated_loss = 0

    # Iterate through the training data
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()  # Zero the gradients
        images, targets = data  # Unpack data
        images = list(image.to(DEVICE) for image in images)  # Move images to device
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # Move targets to device

        # Mixed precision training context
        with amp.autocast():
            loss_dict = model(images, targets)  # Forward pass
            losses = sum(loss for loss in loss_dict.values())  # Compute total loss
            loss_value = losses.item()

        scaler.scale(losses).backward()  # Scale and backpropagate gradients
        
        # Apply gradients every GRADIENT_ACCUMULATION_STEPS steps
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)  # Update optimizer
            scaler.update()  # Update scaler
            optimizer.zero_grad()  # Zero the gradients

        accumulated_loss += loss_value
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
            accumulated_loss = 0  # Reset accumulated loss if not saving gradients

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(images)  # Get model predictions
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_labels = output['labels']
                pred_scores = output['scores']

                gt_boxes = target['boxes']
                gt_labels = target['labels']

                # Calculate precision and recall
                precision, recall = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
                all_precisions.append(precision)
                all_recalls.append(recall)

        model.train()  # Set model back to training mode

        # Calculate mean precision and recall
        mean_precision = torch.tensor(all_precisions, dtype=torch.float).mean().item()
        mean_recall = torch.tensor(all_recalls, dtype=torch.float).mean().item()
        train_loss_hist.send(accumulated_loss)
        train_precision_hist.send(mean_precision)
        train_recall_hist.send(mean_recall)
        
        # Update progress bar description
        prog_bar.set_description(desc=f"Loss: {accumulated_loss:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")

        train_itr += 1

    # Extend lists with collected metrics
    train_precision_list.extend(all_precisions)
    train_recall_list.extend(all_recalls)

    return train_loss_list, train_precision_list, train_recall_list

# Validation function
def validate(valid_data_loader, model):
    print('Validating')  # Print start of validation
    global val_itr  # Use global variable for iteration tracking
    global val_loss_list
    global val_precision_list
    global val_recall_list
    
    # Create a progress bar for validation
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    all_precisions = []
    all_recalls = []
    
    # Iterate through the validation data
    for i, data in enumerate(prog_bar):
        images, targets = data  # Unpack data
        
        model.train()  # Set model to training mode
        
        images = list(image.to(DEVICE) for image in images)  # Move images to device
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]  # Move targets to device
        loss_dict = model(images, targets)  # Forward pass
        losses = sum(loss for loss in loss_dict.values())  # Compute total loss
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(images)  # Get model predictions
        
        for output, target in zip(outputs, targets):
            pred_boxes = output['boxes']
            pred_labels = output['labels']
            pred_scores = output['scores']
            
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            # Calculate precision and recall
            precision, recall = calculate_precision_recall(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        # Calculate mean precision and recall
        mean_precision = torch.tensor(all_precisions, dtype=torch.float).mean().item()
        mean_recall = torch.tensor(all_recalls, dtype=torch.float).mean().item()

        val_loss_hist.send(loss_value)
        val_precision_hist.send(mean_precision)
        val_recall_hist.send(mean_recall)
        
        # Update progress bar description
        prog_bar.set_description(desc=f"Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")
        val_itr += 1
        
    # Extend lists with collected metrics
    val_precision_list.extend(all_precisions)
    val_recall_list.extend(all_recalls)

    return val_loss_list, val_precision_list, val_recall_list

# Function for early stopping based on patience
def early_stopping(metric_values, patience):
    # Check if there is no improvement in the last 'patience' number of epochs
    if len(metric_values) > patience and all(x >= y for x, y in zip(metric_values[-patience:], metric_values[-patience - 1:-1])):
        return True
    return False

# Main execution
if __name__ == '__main__':
    model = create_model(num_classes=NUM_CLASSES)  # Create model with specified number of classes
    model = model.to(DEVICE)  # Move model to device
    
    params = [p for p in model.parameters() if p.requires_grad]  # Filter parameters that require gradients
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)  # Create SGD optimizer
    scaler = amp.GradScaler()  # Create GradScaler for mixed precision training

    # Initialize Averager objects for tracking metrics
    train_loss_hist = Averager()
    train_precision_hist = Averager()
    train_recall_hist = Averager()
    val_loss_hist = Averager()
    val_precision_hist = Averager()
    val_recall_hist = Averager()
    
    train_itr = 1  # Initialize training iteration counter
    val_itr = 1  # Initialize validation iteration counter
    
    # Initialize lists to store metrics
    train_loss_list = []
    train_precision_list = []
    train_recall_list = []
    val_loss_list = []
    val_precision_list = []
    val_recall_list = []
    
    MODEL_NAME = 'model'  # Model name for saving
    best_val_precision = 0  # Initialize best validation precision
    no_improvement_epochs = 0  # Counter for epochs with no improvement
    
    # Visualize transformed images if specified
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image
        show_tranformed_image(train_loader)
    
    # Training loop for each epoch
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")  # Print epoch number
        train_loss_hist.reset()  # Reset training loss history
        val_loss_hist.reset()  # Reset validation loss history
        figure_1, train_ax = plt.subplots()  # Create subplot for training metrics
        figure_2, valid_ax = plt.subplots()  # Create subplot for validation metrics
        start = time.time()  # Record start time
        
        # Train and validate model
        train_loss, train_precision, train_recall = train(train_loader, model, optimizer, scaler)
        val_loss, val_precision, val_recall = validate(valid_loader, model)
        
        # Print metrics for the current epoch
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} train precision: {train_precision_hist.value:.3f}")
        print(f"Epoch #{epoch} train recall: {train_recall_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch} validation precision: {val_precision_hist.value:.4f}")
        print(f"Epoch #{epoch} validation recall: {val_recall_hist.value:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}_{epoch+1}.pth")
        
        # Update best validation precision and check for early stopping
        if val_precision_hist.value > best_val_precision:
            best_val_precision = val_precision_hist.value
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation precision.")
            break

        end = time.time()  # Record end time
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")  # Print epoch duration
        
        # Save plots if specified
        if (epoch+1) % SAVE_MODEL_EPOCH == 0:
            torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}_{epoch+1}.pth")
        
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0:
            train_ax.plot(range(len(train_loss_list)), train_loss_list, label='Train Loss')
            train_ax.plot(range(len(train_precision_list)), train_precision_list, label='Train Precision')
            train_ax.plot(range(len(train_recall_list)), train_recall_list, label='Train Recall')
            train_ax.legend()
            train_ax.set_title('Training Metrics')
            train_ax.set_xlabel('Iterations')
            train_ax.set_ylabel('Metrics')
            figure_1.savefig(f"{OUT_DIR}/train_metrics_{epoch+1}.png")
            
            valid_ax.plot(range(len(val_loss_list)), val_loss_list, label='Validation Loss')
            valid_ax.plot(range(len(val_precision_list)), val_precision_list, label='Validation Precision')
            valid_ax.plot(range(len(val_recall_list)), val_recall_list, label='Validation Recall')
            valid_ax.legend()
            valid_ax.set_title('Validation Metrics')
            valid_ax.set_xlabel('Iterations')
            valid_ax.set_ylabel('Metrics')
            figure_2.savefig(f"{OUT_DIR}/valid_metrics_{epoch+1}.png")

            
