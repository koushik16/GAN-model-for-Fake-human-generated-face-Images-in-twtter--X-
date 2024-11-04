#References:
#1. https://github.com/otenim/Xception-with-Your-Own-Dataset
#2. https://github.com/kooyunmo/GAN-detector
#3. https://github.com/andreacos/gan-generated-face-detection

#libraries
import math
import os
import argparse
import matplotlib
import imghdr
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.optimizers import Adam
import keras.utils as image
from keras.losses import categorical_crossentropy
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import models
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
matplotlib.use('Agg')

#argument from command line: 
#1. arg: dataset main folder (which has two subfolders of real and gan)
#2. arg: classes.txt which has the names of the subfolders of real and gan images.
#3. arg: result directory where the .h5 models after every epoch, the final .h5 and the snaps of the loss and accuracy graphs are saved,

current_directory = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('dataset_root')
parser.add_argument('classes')
parser.add_argument('result_root')
parser.add_argument('--epochs_pre', type=int, default=5)
parser.add_argument('--epochs_fine', type=int, default=10)
parser.add_argument('--batch_size_pre', type=int, default=16)
parser.add_argument('--batch_size_fine', type=int, default=16)
parser.add_argument('--lr_pre', type=float, default=1e-3)
parser.add_argument('--lr_fine', type=float, default=1e-4)
parser.add_argument('--snapshot_period_pre', type=int, default=1)
parser.add_argument('--snapshot_period_fine', type=int, default=1)
parser.add_argument('--split', type=float, default=0.8)

#generating a list of input image paths and corresponding input labels.
def generate_from_paths_and_labels(
        input_paths, labels, batch_size, input_size=(299, 299)):
    num_samples = len(input_paths)
    while 1:
        perm = np.random.permutation(num_samples)
        input_paths = input_paths[perm]
        labels = labels[perm]
        for i in range(0, num_samples, batch_size):
            inputs = list(map(
                lambda x: image.load_img(x, target_size=input_size),
                input_paths[i:i+batch_size]
            ))
            inputs = np.array(list(map(
                lambda x: image.img_to_array(x),
                inputs
            )))
            inputs = preprocess_input(inputs)
            yield (inputs, labels[i:i+batch_size])


def main(args):
    epochs = args.epochs_pre + args.epochs_fine
    args.dataset_root = os.path.expanduser(args.dataset_root)
    args.result_root = os.path.expanduser(args.result_root)
    args.classes = os.path.expanduser(args.classes)

    # loading class names using from classes.txt
    with open(args.classes, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    num_classes = len(classes)

    #loading the dataset from the dataset main folder.
    input_paths, labels = [], []
    for class_name in os.listdir(args.dataset_root):
        class_root = os.path.join(args.dataset_root, class_name)
        class_id = classes.index(class_name)
        for path in os.listdir(class_root):
            path = os.path.join(class_root, path)
            if imghdr.what(path) is None:
                #for a non-image file
                continue
            input_paths.append(path)
            labels.append(class_id)

    #converting labels to categorical.
    labels = to_categorical(labels, num_classes=num_classes)

    #converting to a numpy array
    input_paths = np.array(input_paths)

    #shuffling the dataset
    perm = np.random.permutation(len(input_paths))
    labels = labels[perm]
    input_paths = input_paths[perm]

    #spliting the dataset into train and validation dataset.
    border = int(len(input_paths) * args.split)
    train_labels = labels[:border]
    val_labels_ = labels[border:]
    train_input_paths = input_paths[:border]
    val_input_paths_ = input_paths[border:]

    border_ = int(len(val_input_paths_) * 0.50)
    test_labels = val_labels_[:border_]
    test_input_paths = val_input_paths_[:border]
    val_labels = val_labels_[border_:]
    val_input_paths = val_input_paths_[border_:]

    #print("training on %d images and labels" % (len(train_input_paths)))
    #print("validation on %d images and labels" % (len(val_input_paths)))

    #if a result dir is not mentioned, creating a result directory.
    if os.path.exists(args.result_root) is False:
         os.makedirs(args.result_root)

    # Using a keras pre-trained Xception model, trained on ImageNet weights.
    # the top layer is not included.
    # the input shape is (299, 299, 3)

    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(299, 299, 3))

    #for the top model, we add a global average pooling 2D layer, dense layer and a last softmax layer with dimension of number of classes. 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    #training the model for pre_epochs with pre_learning rate. 5 epochs and 1e-3 learning rate.
    #training the top model, and the basemodel layers are not be trained.
    for layer in base_model.layers:
        layer.trainable = False

    #compiling the top model.
    model.compile(
        loss=categorical_crossentropy,
        optimizer=Adam(lr=args.lr_pre),
        metrics=['accuracy']
    )

    #training the top model.
    hist_pre = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_pre
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / args.batch_size_pre),
        epochs=args.epochs_pre,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_pre
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_pre),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_pre_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_pre,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_pre_final.h5'))

    #Training the model for the fine_epochs with fine_learning rate, 15 epochs with 1e-4 learning rate.
    #here, all the layers are trained.
    for layer in model.layers:
        layer.trainable = True

    #compiling the final model. 
    model.compile(
        optimizer=Adam(lr=args.lr_fine),
        loss=categorical_crossentropy,
        metrics=['accuracy'])

    #training the final model with the total number of epochs. 
    hist_fine = model.fit_generator(
        generator=generate_from_paths_and_labels(
            input_paths=train_input_paths,
            labels=train_labels,
            batch_size=args.batch_size_fine
        ),
        steps_per_epoch=math.ceil(
            len(train_input_paths) / args.batch_size_fine),
        epochs=args.epochs_fine,
        validation_data=generate_from_paths_and_labels(
            input_paths=val_input_paths,
            labels=val_labels,
            batch_size=args.batch_size_fine
        ),
        validation_steps=math.ceil(
            len(val_input_paths) / args.batch_size_fine),
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                filepath=os.path.join(
                    args.result_root,
                    'model_fine_ep{epoch}_valloss{val_loss:.3f}.h5'),
                period=args.snapshot_period_fine,
            ),
        ],
    )
    model.save(os.path.join(args.result_root, 'model_fine_final.h5'))

    #creating the loss and the accuracy graph.
    model = models.load_model('/Users/spetapa/Downloads/XceptionNet/result2/model_pre_final.h5')
    hist_pre = model 
    acc = hist_pre.history['accuracy']
    val_acc = hist_pre.history['val_accuracy']
    loss = hist_pre.history['loss']
    val_loss = hist_pre.history['val_loss']
    acc.extend(hist_fine.history['accuracy'])
    val_acc.extend(hist_fine.history['val_accuracy'])
    loss.extend(hist_fine.history['loss'])
    val_loss.extend(hist_fine.history['val_loss'])

    #saving the graph image
    plt.plot(range(epochs), acc, marker='.', label='accuracy')
    plt.plot(range(epochs), val_acc, marker='.', label='val_accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig(os.path.join(args.result_root, 'accuracy.png'))
    plt.clf()

    plt.plot(range(epochs), loss, marker='.', label='loss')
    plt.plot(range(epochs), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.result_root, 'loss.png'))
    plt.clf()

    #dumping the plot file to a pickle file.
    plot = {
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss,
    }
    with open(os.path.join(args.result_root, 'plot.dump'), 'wb') as f:
        pkl.dump(plot, f)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
