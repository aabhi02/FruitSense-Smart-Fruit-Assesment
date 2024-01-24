import cv2
import numpy as np
from tensorflow.keras.models import load_model
import gradio as gr

# Load the main fruit classification model
main_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/Jan1_sgd_fruit_4layers.h5')

# Load binary classifier models for each fruit
apple_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestAppleModel01.h5')  # Replace with your actual filename
banana_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestBananaModel01.h5')  # Replace with your actual filename
guava_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestGuavaModel01.h5')  # Replace with your actual filename
pomegranate_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestPomegranateModel01.h5')  # Replace with your actual filename
lime_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestLimeModel01.h5')  # Replace with your actual filename
orange_model = load_model('/media/jayanth/Data/Abhi/7th sem/miniProject/savedModel/bestOrangeModel01.h5')  # Replace with your actual filename

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

with gr.Blocks() as demo:
    gr.HTML("<h1><center>FruitSense: Smart Fruit Assesment</center></h1>")
    im = gr.Image()
    txt_3 = gr.Textbox(value="", label="Output")
    btn = gr.Button(value="Submit")
    btn.click(clf, inputs=[im], outputs=[txt_3])

if __name__ == "__main__":
    demo.launch(share=True)

# Example usage
# image_path = 'path/to/your/image.jpg'  # Replace with the actual image path
# fruit_type, ripeness_result = predict_fruit_and_ripeness(image_path)

# print(f"Fruit Type: {fruit_type}")
# print(f"Ripeness: {ripeness_result}")