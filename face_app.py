import streamlit as st
import numpy as np
import cv2
import face_util
import tensorflow as tf
from mtcnn import MTCNN
import openai

page_title = "Face describer"
page_icon = ":male-detective:"
layout = "centered"

image_size = (224, 224)
image_name = "image.jpg"
# TODO remove this key
openai_api_key = "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"

file_types = ["bmp", "dib", "jpeg", "jpg", "jpe", "jp2",
              "png", "webp", "pbm", "pgm", "ppm", "pxm", "pnm",
              "sr", "ras", "tiff", "tif", "exr", "hdr", "pic"]
attributes = ["5_o_Clock_Shadow", "Bald", "Black_Hair", "Blond_Hair", "Brown_Hair",
              "Eyeglasses", "Goatee", "Gray_Hair", "Male", "Mustache", "Smiling", "No_Beard",
              "Wearing_Earrings", "Wearing_Hat"]

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + "" + page_icon)

uploaded_file = st.file_uploader("Upload an image of a face!", type=file_types)


# CV2


@st.cache_resource
def load_model(model_name):
    tr_model = tf.keras.models.load_model(model_name)
    return tr_model


@st.cache_resource
def create_mtcnn_detector():
    detector = MTCNN()
    return detector


@st.cache_resource
def set_open_ai_key():
    openai.api_key = openai_api_key


trained_model = load_model("face_eff_net_all_discarded_bad_samples")
detector = create_mtcnn_detector()
set_open_ai_key()

if uploaded_file is not None:
    img_str = uploaded_file.getvalue()
    st.image(img_str)
    face_util.save_bytes_as_jpg(img_str)
    n_face = face_util.detect_and_save_faces(detector, image_name)
    if n_face > 1:
        st.write(f"There are {n_face} people in this image")
    elif n_face == 1:
        st.write("There is 1 person in this image")
    else:
        st.write("There is no one in this image")
    for i in range(n_face):
        with st.container():
            image = face_util.read_and_prepare_image_as_input(image_size, f"detected_face{i}.jpg")
            st.image(image.numpy())
            prediction_scores = trained_model.predict(np.expand_dims(image.numpy(), axis=0))
            predicted_attributes = []
            for j, label in enumerate(attributes):
                if prediction_scores[j][0][0] > 0.5:
                    predicted_attributes.append(f"{label}")


            person_number = i + 1
            content = f"These are the characteristics of person number {person_number}: {predicted_attributes}. " \
                      f"Describe them. " \
                      f"Refer to the person by their ordinal number e.g., The first person. Start with their gender. " \
                      f"Stick to the provided " \
                      f"characteristics do not make conclusions. Keep it as simple as possible"
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                      messages=[{"role": "user", "content": content}])
            st.write(completion.choices[0].message.content)

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
