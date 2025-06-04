import numpy as np
import tensorflow as tf
from model import build_efficientunet
from utils import preprocess_image, visualize_prediction

def test_model(model_path, image_path):
    """
    Test the trained model on a single image.
    
    Args:
        model_path (str): Path to the saved model
        image_path (str): Path to the test image
    """
    # Load the model
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Make prediction
    pred_mask = model.predict(np.expand_dims(image, axis=0))[0]
    
    # Convert prediction to binary mask
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    
    # For demonstration, we'll create a dummy true mask
    # In practice, you should load the actual ground truth mask
    true_mask = np.zeros_like(pred_mask)
    
    # Visualize the results
    visualize_prediction(image, true_mask, pred_mask)

if __name__ == "__main__":
    # Example usage
    # Replace these paths with your actual model and test image paths
    model_path = "best_model.h5"
    test_image_path = "path/to/your/test/image.jpg"
    
    print("Testing model on sample image...")
    test_model(model_path, test_image_path)
