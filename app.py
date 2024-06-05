import streamlit as st
# import tensorflow as tf
import numpy as np

from PIL import Image
import pickle

model = pickle.load(open("model.pkl",'rb'))

st.title("CIFAR-10 Image Classification")
st.write("Upload an image, and the model will classify it into one of the 10 CIFAR-10 categories.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Load a pre-trained model
classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((32, 32))  # Resizing to match the model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    labe=classes[np.argmax(predictions)]
    #decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
    if st.button('Predict'):
        st.subheader("Predictions:")
        
        #    st.write(f"{i + 1}: {label} ({score:.2f})")
        st.write(predictions)
        st.write(labe)

st.sidebar.markdown("---")
st.sidebar.subheader(" HELLO EVERYONE ")
st.sidebar.text("My Name is Shiv Datta Dixit")
st.sidebar.text('From IIIT BHAGALPUR')
st.sidebar.markdown("[Email:-shivdattadixit0567@gmailc.om]")
st.sidebar.markdown("[Link to GitHub:-https://github.com/shivdattadixit0567]")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.write("This model is trained on the CIFAR10 dataset and can be used to classify images into 10 categories.")
# Optional: Add a link to your GitHub repository or any additional information
st.sidebar.markdown("---")

# Optional: Add a footer

# Run the app
if __name__ == "__main__":
    st.markdown("---")
    st.write('These are the 10 classes of CIFAR10 dataset')
    st.write(classes)
    st.markdown("### Instructions")
    st.markdown("1. Upload an image.")
    st.markdown("2. The model will classify the image into one of the CIFAR-10 categories mentioned above.")

