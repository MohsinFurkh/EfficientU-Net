import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize
    image = cv2.resize(image, (target_size[1], target_size[0]))
    image = image / 255.0
    
    return image.astype(np.float32)

def preprocess_mask(mask_path, target_size=(256, 256)):
    """
    Load and preprocess a mask image.
    
    Args:
        mask_path (str): Path to the mask file
        target_size (tuple): Target size for resizing (height, width)
        
    Returns:
        numpy.ndarray: Preprocessed mask
    """
    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize and binarize
    mask = cv2.resize(mask, (target_size[1], target_size[0]))
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = mask / 255.0
    
    return np.expand_dims(mask, axis=-1).astype(np.float32)

def visualize_prediction(image, true_mask, pred_mask, alpha=0.5):
    """
    Visualize the original image, ground truth mask, and predicted mask.
    
    Args:
        image (numpy.ndarray): Input image
        true_mask (numpy.ndarray): Ground truth mask
        pred_mask (numpy.ndarray): Predicted mask
        alpha (float): Transparency for mask overlay
    """
    plt.figure(figsize=(15, 5))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(true_mask.squeeze(), alpha=alpha, cmap='jet')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Plot predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(pred_mask.squeeze(), alpha=alpha, cmap='jet')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    Args:
        history: Keras History object returned from model.fit()
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()
