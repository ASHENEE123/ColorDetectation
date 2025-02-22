
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ✅ Load the trained CNN model
model_path = "new_model.h5"  # Change this path if needed
model = tf.keras.models.load_model(model_path)

# ✅ Get input image dimensions
img_height, img_width = model.input_shape[1], model.input_shape[2]
is_grayscale = model.input_shape[3] == 1  # Check if the model expects grayscale images

# ✅ Streamlit UI
st.title("🖼️ Image Classification with CNN")
st.write("Upload an image and get a prediction with confidence score.")

# ✅ Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Convert image to grayscale if required by model
    if is_grayscale:
        image = image.convert("L")

    # ✅ Resize image to match model's input shape
    image = image.resize((img_width, img_height))

    # ✅ Convert image to numpy array and normalize
    image_array = np.array(image) / 255.0  # Normalize pixel values
    
    # ✅ Reshape if grayscale (add channel dimension)
    if is_grayscale:
        image_array = np.expand_dims(image_array, axis=-1)

    # ✅ Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ✅ Make Prediction
    predictions = model.predict(image_array)
    prediction = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))

    # ✅ Show Prediction Result
    st.subheader("🧠 Prediction Result")
    st.write(f"**Prediction:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}")
