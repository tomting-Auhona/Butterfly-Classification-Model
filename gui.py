import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model = load_model("butterfly.h5")

# Define the scientific names of the butterfly categories
butterfly_categories = {
    1: "Danaus plexippus",
    2: "Heliconius charitonius",
    3: "Heliconius erato",
    4: "Junonia coenia",
    5: "Lycaena phlaeas",
    6: "Nymphalis antiopa",
    7: "Papilio cresphontes",
    8: "Pieris rapae",
    9: "Vanessa atalanta",
    10: "Vanessa cardui"
}

# Description for each butterfly species
butterfly_descriptions = {
    1: "Danaus plexippus, commonly known as the monarch butterfly, "
       "is well-known for its distinctive orange and black patterned wings."
       "(89-102 mm). Very large, with FW long and drawn out. Above, bright, "
       "burnt-orange with black veins and black margins sprinkled with white dots; "
       "FW tip broadly black interrupted by larger white and orange spots. Below, paler, duskier orange. "
       "1 black spot appears between HW cell and margin on male above and below. "
       "Female darker with black veins smudged.",

    2: "Heliconius charitonius, also called the zebra longwing, is recognized by its long, narrow wings with "
       "black and yellow stripes. (76-78 mm). Wings long and narrow. Jet-black above, "
       "banded with lemon-yellow (sometimes pale yellow). "
       "Beneath similar; bases of wings have crimson spots.",

    3: "Heliconius erato, known as the red postman, features vibrant red, black, "
       "and yellow markings on its wings. (76-86 mm). Wings long, narrow, and rounded. Black above, "
       "crossed on FW by broad crimson patch, and on HW by narrow yellow line. "
       "Below, similar but red is pinkish and HW has less yellow.",

    4: "Junonia coenia, or the common buckeye, has eye-like markings on its wings and is commonly found in "
       "North and Central America. (51-63 mm). Wings scalloped and rounded except at drawn-out FW tip. "
       "Highly variable. Above, tawny-brown to dark brown; 2 orange bars in FW cell, orange submarginal band on "
       "HW, white band diagonally crossing FW. 2 bright eyespots on each wing above: on FW, 1 very small near "
       "tip and 1 large eyespot in white FW bar; on HW, 1 large eyespot near upper margin and 1 small eyespot "
       "below it. Eyespots black, yellow-rimmed, with iridescent blue and lilac irises. Beneath, FW resembles "
       "above in lighter shades; HW eyespots tiny or absent, rose-brown to tan, with vague crescent-shaped markings ",

    5: "Lycaena phlaeas, the small copper butterfly, is characterized by its bright orange wings with black spots. "
       "(22-28 mm). Above, FW bright copper or brass-colored with dark spots and margin; HW dark brown with copper "
       "margin. "
       "Undersides mostly grayish with black dots; FW has some orange, HW has prominent submarginal orange band.",

    6: "Nymphalis antiopa, or the mourning cloak, has dark brown wings with a distinctive border of blue spots "
       "and a yellow edge. "
       "(73-86 mm). Large. Wing margins ragged. Dark with pale margins. Above, rich brownish-maroon, "
       "iridescent at close range, with ragged, cream-yellow band, bordered inwardly by brilliant blue spots "
       "all along both wings. Below, striated, ash-black with row of blue-green to blue-gray chevrons just inside "
       "dirty yellow border.",

    7: "Papilio cresphontes, the giant swallowtail, is one of the largest butterflies in North America, "
       "featuring yellow and black wings. "
       "(86-140 mm). Very large. Long, dark, spoon-shaped tails have yellow center. "
       "Dark brownish-black above with 2 broad bands of yellow spots converging at tip of FW. "
       "Orange spot at corner of HW flanked by blue spot above; both recur below, but blue continuing in "
       "chevrons across underwing, which also has orange patch. Otherwise, yellow below with black veins and "
       "borders. Abdomen yellow with broad black midline tapering at tip; notch on top of abdomen near rear. "
       "Thorax has yellow lengthwise spots or stripes.",

    8: "Pieris rapae, known as the small cabbage white, has white wings with black markings and is commonly "
       "found in gardens and fields. (32-48 mm). Milk-white above with charcoal "
       "spots on FW (1 on male, 2 on female) and on HW costa. "
       "Below, FW tip and HW pale to bright mustard-yellow, speckled with grayish spots and black FW spots.",

    9: "Vanessa atalanta, or the red admiral, has black wings with orange bands and white spots along the edges."
       "(44-57 mm). FW tip extended, clipped. Above, black with orange-red to vermilion bars across FW and on HW "
       "border. Below, mottled black, brown, and blue with pink bar on FW. "
       "White spots at FW tip above and below, bright blue patch on lower HW angle above and below.",

    10: "Vanessa cardui, the painted lady butterfly, is known for its orange, black, and white wings with "
        "distinctive black eye-spots. (51-57 mm). FW tip extended slightly, rounded. "
        "Above, salmon-orange with black blotches, black-patterned margins, and broadly black FW tips with "
        "clear white spots; outer HW crossed by small black-rimmed blue spots. Below, FW dominantly rose-pink "
        "with olive, black, and white pattern; HW has small blue spots on olive background with white webwork. "
        "FW above and below has white bar"
}


# Function to preprocess the image
def preprocess_image(image):
    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to match model input size
    image = cv2.resize(image, (224, 224))
    # Normalize pixel values
    image = image / 255.0
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)
    return image


# Function to make predictions
def predict_butterfly(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(processed_image)
    # Get the predicted butterfly category
    predicted_category = np.argmax(prediction) + 1  # Adjusted index to start from 1
    return predicted_category


# Create Tkinter window
window = tk.Tk()
window.title("Butterfly Species Prediction")
window.geometry("600x600")

# Add background image
background_image = Image.open("background_image.jpg")
background_image = background_image.resize((600, 600), Image.ANTIALIAS)
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Heading label
heading_label = tk.Label(window, text="Butterfly Species Prediction", font=("Comic Sans MS", 20), bg="white", fg="#FD0542")
heading_label.place(x=150, y=50)

# Label to display prediction
prediction_label = tk.Label(window, text="", font=("Comic Sans MS", 14), bg="#E2EAF4")
prediction_label.place(x=135, y=320)

# Label to display description
description_label = tk.Label(window, text="", font=("Comic Sans MS", 10), wraplength=300, justify="left", bg="#E8E8E8")
description_label.place(x=165, y=350)

# Function to select image and predict
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        predicted_category = predict_butterfly(image)
        prediction_label.config(text=f"Predicted Butterfly Species: {butterfly_categories[predicted_category]}")
        description_label.config(text=f"Description:\n{butterfly_descriptions[predicted_category]}")

# Button to select image
select_button = tk.Button(window, text="Upload Image", command=select_image, font=("Comic Sans MS", 14), bg="red", fg="white")
select_button.place(x=250, y=100)

# Run the Tkinter event loop
window.mainloop()
