import streamlit as st
import face_util_tflite
import tflite_runtime.interpreter as tflite
import openai
import json

secret_name = "Openai_api_key"
region_name = "us-east-1"

page_title = "Face describer"
page_icon = ":male-detective:"
layout = "centered"

image_size = (224, 224)
image_name = "image.jpg"

file_types = ["bmp", "dib", "jpeg", "jpg", "jpe", "jp2",
              "png", "webp", "pbm", "pgm", "ppm", "pxm", "pnm",
              "sr", "ras", "tiff", "tif", "exr", "hdr", "pic"]
attributes = ["5_o_Clock_Shadow", "Bald", "Black_Hair", "Blond_Hair", "Brown_Hair",
              "Eyeglasses", "Goatee", "Gray_Hair", "Male", "Mustache", "Smiling", "No_Beard",
              "Wearing_Earrings", "Wearing_Hat"]

model_name = "my_model.tflite"

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + "" + page_icon)

with st.container():
    st.write("""
    **Please upload an image showing the face of exactly one person. The application detects the faces in the image and describes them.**
    
    """)

uploaded_file = st.file_uploader("Upload image", type=file_types)





@st.cache_resource
def set_open_ai_key():
    return face_util_tflite.get_openai_api_key(secret_name, region_name)



@st.cache_resource
def create_tflite_interpreter():
    interpreter = tflite.Interpreter(model_path=model_name)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'],
                                    (1, input_details[0]['shape'][1], input_details[0]['shape'][2], 3))
    return interpreter, input_details, output_details


openai.api_key = set_open_ai_key()
interpreter, input_details, output_details = create_tflite_interpreter()

if uploaded_file is not None:
    n_face = 1
    img_str = uploaded_file.getvalue()
    st.image(img_str)
    face_util_tflite.save_bytes_as_jpg(img_str)
    # n_face = face_util_tflite.detect_and_save_faces(detector, image_name)
    # if n_face > 1:
    #     st.write(f"There are {n_face} people in this image")
    # elif n_face == 1:
    #     st.write("There is 1 person in this image")
    # else:
    #     st.write("There is no one in this image")

    face_descriptions_dict = dict()

    for i in range(n_face):
        image = face_util_tflite.read_and_prepare_image_as_input(image_size, f"image.jpg")
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        sorted_output_details = sorted(output_details, key=lambda x: x['index'])
        output_data = [interpreter.get_tensor(sorted_output_details[i]['index']) for i in range(len(output_details))]
        prediction_scores = [arr[0][0] for arr in output_data]
        predicted_attributes = []
        for j, label in enumerate(attributes):
            if prediction_scores[j] > 0.5:
                predicted_attributes.append(f"{label}")

        if "Male" not in predicted_attributes:
            predicted_attributes.append("Female")
        if "No_Beard" in predicted_attributes:
            predicted_attributes.remove("No_Beard")
        elif all(elem not in predicted_attributes for elem in ["Goatee", "5_o_Clock_Shadow", "Mustache"]):
            predicted_attributes.append("Beard")

        face_descriptions_dict[f"person{i + 1}"] = predicted_attributes

    content = f"These are the characteristics of {n_face} people: {face_descriptions_dict}. " \
              f"Describe them using full sentences. " \
              f"If there are more than one person, refer to them by their ordinal number e.g (The first person), " \
              f"otherwise refer to the person as 'the person'. Always start with their genders. " \
              f"Stick to the provided characteristics do not make conclusions. Keep it as simple as possible. " \
              f"Return the response in JSON format. The keys of this dictionary must be person numbers (e.g., 1)" \
              f"The values of the dictionary must be the corresponding person's description. " \
              f"values." \
              f"Put keys and values in double quotes."

    indexes = set([str(i) for i in range(1, n_face + 1)])
    try_describing = True
    response_dict = {}
    while try_describing:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "user", "content": content}])
        response_dict = json.loads(completion.choices[0].message.content)
        if set(response_dict.keys()) == indexes:
            try_describing = False

    for i in range(n_face):
        with st.container():
            image = face_util_tflite.read_and_prepare_image_as_input(image_size, "image.jpg")
            st.image(image)
            st.write(response_dict[f"{i + 1}"])
