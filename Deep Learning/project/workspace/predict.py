import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
from process_image import process_image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("x")
parser.add_argument("y")
parser.add_argument("--top_k",  type=int, required=False,default=3)
parser.add_argument("--category_names",  required=False,default = 0)
args = vars(parser.parse_args())
im = Image.open(args['x'])
test_image = np.asarray(im)

processed_test_image = process_image(test_image)
processed_test_image=np.expand_dims(processed_test_image, axis=0)
topk=args['top_k']
model = tf.keras.models.load_model(args['y'],custom_objects={'KerasLayer':hub.KerasLayer})
result=model.predict(processed_test_image)
cls=tf.math.top_k(result,topk)[1].numpy()[0]

if(args['category_names']):
    with open(args['category_names'], 'r') as f:
        class_names = json.load(f)
    print(class_names)
else:
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
lists = [] 
for key in class_names.keys():
    lists.append(int(key))
lists.sort()
classes=[]
for i in cls:
    classes.append(class_names[str(lists[i])])
print(classes)
