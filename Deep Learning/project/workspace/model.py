import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
def model(network,processed_test_image,filename,topk):
    model = tf.keras.models.load_model(network,custom_objects={'KerasLayer':hub.KerasLayer})
    result=model.predict(processed_test_image)
    cls=tf.math.top_k(result,topk)[1].numpy()[0]
    prob=tf.math.top_k(result,topk)[0].numpy()[0]
   
    with open(filename, 'r') as f:
        class_names = json.load(f)
    classes=[]
    for i in cls:
        classes.append(class_names[str(i+1)])
    probs=[]
    for i in prob:
        probs.append(i)
    return probs,classes
