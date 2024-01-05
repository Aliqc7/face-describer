FROM python:3.10-slim

WORKDIR /application

COPY tflite_requirements.txt .

RUN pip install -r tflite_requirements.txt

COPY App_tflite .

CMD ["streamlit", "run" , "./face_app_tflite.py"]