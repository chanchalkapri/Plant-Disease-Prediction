from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi.responses import StreamingResponse

app = FastAPI()

# Load models
model = tf.keras.models.load_model("../models/3.h5")
VGG_MODEL = VGG16(weights='imagenet', include_top=True)

# Get the last convolutional layer of VGG16
last_conv_layer = VGG_MODEL.get_layer('block5_conv3')
grad_model = tf.keras.models.Model([VGG_MODEL.inputs], [last_conv_layer.output, VGG_MODEL.output])

# Define class names
CLASS_NAMES = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot", "Spotted Spider Mite", "Target Spot", "Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy"]

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((224, 224)))
    return image

@app.post("/predict")
async def predict_gradcam(file: UploadFile = File(...)):


    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    # Predict disease

    predictions = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

        
    
    x = preprocess_input(img_batch)
    
    
    # Generate Grad-CAM heatmap
    class_idx = np.argmax(predictions[0])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        class_output = preds[:, class_idx]
    grads = tape.gradient(class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = cv2.resize(np.float32(heatmap), (x.shape[2], x.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype(np.float32)
    
    superimposed_img = cv2.addWeighted(x[0], 0.6, heatmap, 0.4, 0)
    result_image = Image.fromarray(np.uint8(superimposed_img))
    
    img_io = BytesIO()
    result_image.save(img_io, format='JPEG')
    img_io.seek(0)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "image": img_io.getvalue().hex()  # Send image as hex string
    }

if __name__ == "__main2__":
    uvicorn.run(app, host='localhost', port=8000)
