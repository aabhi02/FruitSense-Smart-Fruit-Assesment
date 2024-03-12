import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import streamlit as st

# Load the main fruit classification model
main_model = load_model('./models/Jan1_sgd_fruit_4layers.h5')

# Load binary classifier models for each fruit
apple_model = load_model('./models/bestAppleModel01.h5')  # Replace with your actual filename
banana_model = load_model('./models/bestBananaModel01.h5')  # Replace with your actual filename
guava_model = load_model('./models/bestGuavaModel01.h5')  # Replace with your actual filename
pomegranate_model = load_model('./models/bestPomegranateModel01.h5')  # Replace with your actual filename
lime_model = load_model('./models/bestLimeModel01.h5')  # Replace with your actual filename
orange_model = load_model('./models/bestOrangeModel01.h5')  # Replace with your actual filename

def clf(img):
    img = cv2.resize(img, (256, 256))  # Ensure the image size matches your model input size
    img = img / 255.0  # Normalize pixel values

    # Expand dimensions to match the model input shape (add batch dimension)
    img = np.expand_dims(img, axis=0)

    # Predict the main fruit type using the main model
    main_prediction = main_model.predict(img)

    # Get the predicted class index
    main_class_index = np.argmax(main_prediction)

    # Choose the binary classifier model based on the predicted fruit type
    if main_class_index == 0:  # Assuming Apple is the first class
        ripeness_prediction = apple_model.predict(img)
        fruit_name = "Apple"
    elif main_class_index == 1:  # Assuming Banana is the second class
        ripeness_prediction = banana_model.predict(img)
        fruit_name = "Banana"
    elif main_class_index == 2:  # Assuming Guava is the third class
        ripeness_prediction = guava_model.predict(img)
        fruit_name = "Guava"
    elif main_class_index == 3:  # Assuming Lime is the fourth class
        ripeness_prediction = lime_model.predict(img)
        fruit_name = "Lime"
    elif main_class_index == 4:  # Assuming Orange is the fifth class
        ripeness_prediction = orange_model.predict(img)
        fruit_name = "Orange"
    elif main_class_index == 5:  # Assuming Pomegranate is the sixth class
        ripeness_prediction = pomegranate_model.predict(img)
        fruit_name = "Pomegranate"
    else:
        return "Unknown Fruit", None


    # Get the binary classifier's prediction result
    ripeness_result = "Ripe" if np.argmax(ripeness_prediction) == 1 else "Rotten"

    return fruit_name + " " + ripeness_result



if __name__ == "__main__":

    st.title("FruitSense: Smart Fruit Assesment")
    st.markdown("""Currently the project supports 6 categories of fruits:\n
1. Apple\n
2. Banana\n
3. Guava\n
4. Lime\n
5. Orange\n
6. Pomegranate""")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the uploaded file to a PIL image
        image_bytes = io.BytesIO(uploaded_file.read())
        pil_image = Image.open(image_bytes)

        # Display the uploaded image
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Button to submit the image for processing
        if st.button("Submit"):
            # Call the function to process the PIL image
            result = clf(np.array(pil_image))
            # Display the result
            st.markdown(f"## {result.split()[0]} is {result.split()[1]}")
