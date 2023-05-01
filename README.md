# face-describer

Identifies faces in an uploaded image and describes them. 

Link to app: http://face.eizee.xyz

**Table of Contetnts**
- [Description](#description)
- [Execution](#execution)
- [Model training](#Model training)

### **Description:**

This application, written in Python, identifies images in an uploaded image and describes them. 
The faces in the image are identified using **MTCNN** Python package. The attributes of the identified
faces are recognised using an **Efficient Net** NN model trained on more than 129,000 celebrity faces. The UI 
for the app is created using **Streamlit** Python package. 

### Execution

To run the app locally execute the following command in the terminal:
```
streamlit run face_app.py
```
This requires installing the dependencies from the requirements.txt file:
```
pip install -r requirements.txt
```

### Model training

The details of data preperation, model creation, and training can be found in the note book 
**face_describer_model_training.ipynb**


