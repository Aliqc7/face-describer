import cv2
from PIL import Image
import numpy as np
import boto3
from botocore.exceptions import ClientError
import ast
import math
from pathlib import Path


def image_to_jpg(image_name):
    name, ext = image_name.split(".")
    image = cv2.imread(image_name)
    cv2.imwrite(f"{name}.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# def read_and_prepare_image_as_input(image_size, image_name):
#     image = tf.io.read_file(image_name)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, image_size)
#     normalization_layer = tf.keras.layers.Rescaling(1. / 255)
#     image = normalization_layer(image)
#     return image
#
#
# def prepare_image(image, image_size):
#     image = tf.image.resize(image, image_size)
#     normalization_layer = tf.keras.layers.Rescaling(1. / 255)
#     image = normalization_layer(image)
#     return image


def read_and_prepare_image_as_input(image_size, image_name):
    image = Image.open(image_name)
    image = image.resize(image_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


def save_bytes_as_jpg(img_str):
    np_arr = np.fromstring(img_str, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite('image.jpg', img_np)


def detect_and_save_faces(detector, image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    original_img = cv2.imread(image_path)
    detected_dict = detector.detect_faces(img)
    for i, detected in enumerate(detected_dict):
        box = detected["box"]
        temp_box = box
        box[0] = max(temp_box[0] - math.floor(0.5 * temp_box[2]), 0)
        box[1] = max(temp_box[1] - math.floor(0.5 * temp_box[3]), 0)
        box[2] = math.floor(2 * temp_box[2])
        box[3] = math.floor(2 * temp_box[3])
        roi = original_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        cv2.imwrite(f"detected_face{i}.jpg", roi)
    return len(detected_dict)


def get_openai_api_key(secret_name, region_name):
    secret_name = secret_name
    region_name = region_name

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    secret_dict = ast.literal_eval(secret)
    return next(iter(secret_dict))


def copy_model_from_s3(model_name):

    if not Path(model_name).is_dir():
        s3 = boto3.resource('s3')
        bucket_name = 'face-effnet-model'
        prefix = f'{model_name}/'

        # List all objects in the S3 bucket
        bucket = s3.Bucket(bucket_name)

        for elem in bucket.objects.filter(Prefix=prefix):
            key = elem.key
            s = key.split('/')
            dir = '/'.join(s[:-1])
            file = s[-1]
            Path(dir).mkdir(parents=True, exist_ok=True)
            if file:
                bucket.download_file(key, key)