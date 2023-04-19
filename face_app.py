import streamlit as st
import numpy as np
import cv2
import face_util
import tensorflow as tf

page_title = "Face describer"
page_icon = ":male-detective:"
layout = "centered"

image_size = (224, 224)
image_name = "image.jpg"

file_types = ["bmp", "dib", "jpeg", "jpg", "jpe", "jp2",
              "png", "webp", "pbm", "pgm", "ppm", "pxm", "pnm",
              "sr", "ras", "tiff", "tif", "exr", "hdr", "pic"]
attributes = ["5_o_Clock_Shadow", "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
              "Eyeglasses", "Goatee", "Gray_Hair", "Male", "Mustache", "No_Beard", "Sideburns", "Smiling",
              "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
              "Wearing_Necktie"]

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + "" + page_icon)

uploaded_file = st.file_uploader("Upload an image of a face!", type=file_types)


# CV2


@st.cache_resource
def load_model(model_name):
    tr_model = tf.keras.models.load_model(model_name)
    return tr_model


trained_model = load_model("face_eff_net_all_samples")

if uploaded_file is not None:
    img_str = uploaded_file.getvalue()
    st.image(img_str)
    face_util.save_bytes_as_jpg(img_str)
    image = face_util.read_and_prepare_image_as_input(image_size, image_name)
    st.image(image.numpy())
    # np_arr = np.fromstring(img_str, np.uint8)
    # image_tf = tf.constant(np_arr)
    # image = tf.image.resize(image_tf, image_size)
    # normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    # image = normalization_layer(image)
    #
    #
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # image = face_util.prepare_image(img_np, image_size)
    # st.image(img_np)
    prediction_scores = trained_model.predict(np.expand_dims(image.numpy(), axis=0))
    predicted_attributes = []
    for i, label in enumerate(attributes):
        if prediction_scores[i][0][0] > 0.5:
            predicted_attributes.append(f"{label}")
        else:
            predicted_attributes.append(f" not {label}")
    st.write(predicted_attributes)

