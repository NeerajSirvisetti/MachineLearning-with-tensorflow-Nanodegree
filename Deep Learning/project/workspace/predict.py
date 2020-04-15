import argparse
from model import model
from PIL import Image
from process_image import process_image
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("x")
parser.add_argument("y")
parser.add_argument("--top_k",  type=int, required=False,default=1)
parser.add_argument("--category_names",  required=False,default = 'label_map.json')
args = vars(parser.parse_args())
im = Image.open(args['x'])
test_image = np.asarray(im)

processed_test_image = process_image(test_image)
processed_test_image=np.expand_dims(processed_test_image, axis=0)

topk=args['top_k']
probs,classes=model(args['y'],processed_test_image,args['category_names'],topk)

for i in range(len(probs)):
    print(classes[i],":",probs[i])

