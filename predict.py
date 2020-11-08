#import libriaries
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
tfds.disable_progress_bar()
import json

parser = argparse.ArgumentParser(description='Image Classifier - Prediction Part')
parser.add_argument('--input', default='./flowers/test/1/image_06752.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./model.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
top_k = arg_parser.top_k
category_names = arg_parser.category_names


def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(image, (224,224))
    img = img/255
    img  = img.numpy()
    return img

def predict(image_path, model, top_k=5):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    transform_img = process_image(test_img)
    predicton= model.predict(np.expand_dims(transform_img , axis=0))
    predicton = predicton.tolist()
    
    probs, classes = tf.math.top_k(predicton, k=top_k)
    probs=probs.numpy().tolist()[0]
    classes= classes.numpy().tolist()[0]
    return probs, classes


if __name__ == "__main__":
       
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    model_2 = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    
    probs, classes = predict(image_path, model_2, top_k=5)
    
    label_names = [class_names[str(idd)] for idd in classes]
    predict(image_path, model, top_k=5)
    print(probs)
    print(classes)
    print(label_names)
