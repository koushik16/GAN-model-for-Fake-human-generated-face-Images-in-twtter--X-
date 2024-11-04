import glob
import numpy as np
from keras.applications.xception import preprocess_input
from keras import models
from keras.preprocessing.image import ImageDataGeneratorstyle_gan2 
import keras.utils as image
from keras.models import load_model
import pandas as pd

style_gan2 = glob.glob('/Users/spetapa/Downloads/000000/*.png')
total = 1000

def main():

    #creating model
    model = models.load_model('/Users/spetapa/Downloads/XceptionNet/result1/model_fine_final.h5')

    classes = []
    with open(f'classes.txt', 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

    count = 0
    # load an input image
    for img in style_gan2 :
        img_ = image.load_img(img, target_size=(299, 299))
        x = image.img_to_array(img_)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        result=[]
        # predict
        pred = model.predict(x)[0]
        result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
        result.sort(reverse=True, key=lambda x: x[1])
        #print(result)
        
        class_name, prob = result[0]
        #print(class_name)
        if class_name == 'GAN':
            count+=1

    print("accuracy: ",count/total)


if __name__ == '__main__':
    main()
