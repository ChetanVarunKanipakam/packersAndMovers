import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib

# Load your trained Keras model
model = joblib.load("image_classification_model.pkl")  # <-- Check filename!

# Define class names
class_names = ['almirah', 'chair', 'table', 'table', 'tv']
dic = {'almirah': 15, 'chair': 10, 'fridge': 20, 'table': 25, 'tv': 10}

# Define location labels
location_labels = [
    'Amaravathi', 'Ananthapur', 'Bengaluru', 'Chennai', 'Delhi', 'Hyderabad', 
    'Idupulapaya', 'Kadapa', 'Kurnool', 'Mumbai', 'Nellore', 'Ongole', 
    'Produttur', 'Pune', 'Tirupati', 'Vempalli', 'Vijayawada'
]

# Distances between locations
distances = [
    [0, 382, 516, 457, 1670, 269, 289, 283, 346, 973, 335, 184, 367, 881, 418, 292, 63],
    [382, 0, 212, 385, 1862, 354, 99, 111, 112, 984, 379, 478, 102, 770, 329, 86, 460],
    [516, 212, 0, 346, 2154, 569, 277, 266, 274, 981, 379, 467, 239, 841, 252, 238, 579],
    [457, 385, 346, 0, 2196, 627, 409, 398, 458, 1260, 177, 300, 376, 915, 135, 382, 438],
    [1670, 1862, 2154, 2196, 0, 1550, 1688, 1675, 1616, 1415, 1797, 1880, 1696, 1425, 1980, 1690, 1676],
    [269, 354, 569, 627, 1550, 0, 246, 240, 218, 712, 453, 306, 399, 649, 525, 253, 303],
    [289, 99, 277, 409, 1688, 246, 0, 14, 76, 885, 328, 428, 30, 695, 358, 10, 349],
    [283, 111, 266, 398, 1675, 240, 14, 0, 64, 872, 319, 419, 23, 682, 347, 6, 343],
    [346, 112, 274, 458, 1616, 218, 76, 64, 0, 844, 325, 412, 87, 653, 405, 66, 402],
    [973, 984, 981, 1260, 1415, 712, 885, 872, 844, 0, 1240, 1333, 906, 148, 1215, 878, 1036],
    [335, 379, 379, 177, 1797, 453, 328, 319, 325, 1240, 0, 118, 325, 1091, 136, 323, 308],
    [184, 478, 467, 300, 1880, 306, 428, 419, 412, 1333, 118, 0, 414, 1176, 270, 415, 191],
    [367, 102, 239, 376, 1696, 399, 30, 23, 87, 906, 325, 414, 0, 719, 367, 34, 370],
    [881, 770, 841, 915, 1425, 649, 695, 682, 653, 148, 1091, 1176, 719, 0, 1062, 688, 946],
    [418, 329, 252, 135, 1980, 525, 358, 347, 405, 1215, 136, 270, 367, 1062, 0, 356, 376],
    [292, 86, 238, 382, 1690, 253, 10, 6, 66, 878, 323, 415, 34, 688, 356, 0, 346],
    [63, 460, 579, 438, 1676, 303, 349, 343, 402, 1036, 308, 191, 370, 946, 376, 346, 0]
]

# Styling - Background and Color
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://images.unsplash.com/photo-1604147706284-943bde66edb1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1770&q=80");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

[data-testid="stHeader"] {{
    background-color: rgba(0,0,0,0);
}}

h1, h2, h3, h4 {{
    color: #f1f1f1;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-color: #111;
}}

.stButton>button {{
    background-color: #0099ff;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px 24px;
}}

.stSelectbox>div>div>div>div {{
    background-color: white;
    color: black;
}}

div.stAlert {{
    background-color: #f0f2f6;
    border-left: 5px solid #0099ff;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Function to preprocess and predict
def predict_image(model, img, class_names, img_height=128, img_width=128):
    img = img.resize((img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return predicted_class, confidence

def main():
    st.title("🚛 Packers and Movers 🚛")
    st.subheader("Upload an image of the object you want to move 📦")

    # Upload Image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        with st.spinner('Predicting...'):
            predicted_class, confidence = predict_image(model, img, class_names)

        st.info(f"Confidence Score: {confidence:.2f}")

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        st.pyplot(fig)

    st.divider()

    # Select Locations
    st.header("📍 Select Locations for Transport")

    with st.form("location_form"):
        location1 = st.selectbox("Select Pickup Location", location_labels)
        remaining_locations = [loc for loc in location_labels if loc != location1]
        location2 = st.selectbox("Select Drop Location", remaining_locations)

        submitted = st.form_submit_button("Calculate Cost 🚚")

    if submitted:
        idx1 = location_labels.index(location1)
        idx2 = location_labels.index(location2)
        if idx1 == idx2:
            st.warning("Please select two different locations.")
        else:
            cost = int(distances[idx1][idx2] * dic[class_names[predicted_class]])
            st.success(f"Estimated Transportation Cost: ₹{cost}")

if __name__ == "__main__":
    main()
