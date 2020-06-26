import IPython.display as display
from PIL import Image, ImageOps
import os
import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from unet import *
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

datasetUrlTrain = '...' # dataset url to 2018 Data science bowl data
datasetUrlTest = '...' # dataset url to 2018 Data science bowl data

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCHSIZE = 32
SHUFFLE_BUFFER_SIZE = 1500


train_data_dir = pathlib.Path(datasetUrlTrain)
test_data_dir = pathlib.Path(datasetUrlTest)


# Extract Image, Mask pairs if train, else Image. 
def decode_img(file_path, is_label=False):

    img = tf.io.read_file(file_path)    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if is_label == True:
        img = tf.image.rgb_to_grayscale(img)
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])

def get_mask(file_path):

    masks_path = tf.strings.join([file_path, "/masks/"]).numpy()
    mask_img = tf.constant(0)
    for ind, fl in enumerate(tf.io.gfile.listdir(masks_path)):
        
        path = tf.strings.join([masks_path,fl], separator="/")
        if ind == 0:
            mask_img = decode_img(path, is_label= True)
        else:
            mask_img = mask_img + decode_img(path, is_label= True)
    mask_img = tf.cast(mask_img > 0.5, tf.float32)

    return mask_img

def process_path(file_path, is_test=False):

    parts = tf.strings.split(file_path, os.path.sep) 
    img_path = tf.strings.join([file_path, "/images/", parts[-1], ".png"])
    img = decode_img(img_path)
    
    if is_test :
        return img
        
    mask_img = tf.py_function(get_mask, [file_path], Tout=(tf.float32))

    return img, mask_img

# Random brightness augmentation
def augment_brightness(img, label):
    return tf.image.random_brightness(img, 0.5), label
# 90 degree rotation
def augment1(img, label):
    return tf.image.rot90(img), tf.image.rot90(label)
# Flip rotation
def augment2(img, label):
    return  tf.image.flip_left_right(img),  tf.image.flip_left_right(label)

AUTOTUNE = tf.data.experimental.AUTOTUNE
labeled_ds = list_train_ds.map(process_path)
labeled_ds = labeled_ds.shuffle(buffer_size= SHUFFLE_BUFFER_SIZE)

TRAIN_SAMPLES = 500

augment1_train_ds = labeled_ds.take(TRAIN_SAMPLES).cache().map(augment1).batch(BATCHSIZE).prefetch(buffer_size=AUTOTUNE)
augment2_train_ds = labeled_ds.take(TRAIN_SAMPLES).cache().map(augment_brightness).batch(BATCHSIZE).prefetch(buffer_size=AUTOTUNE)
labeled_train_ds = labeled_ds.take(TRAIN_SAMPLES).cache().batch(BATCHSIZE).prefetch(buffer_size=AUTOTUNE)

labeled_val_ds = labeled_ds.skip(TRAIN_SAMPLES).cache()
labeled_val_ds = labeled_val_ds.batch(BATCHSIZE)
labeled_val_ds = labeled_val_ds.prefetch(buffer_size=AUTOTUNE)

test_ds = list_test_ds.map(lambda x : process_path(x, True)).batch(BATCHSIZE).prefetch(buffer_size= AUTOTUNE)

tf.random.set_seed(5)

model1 = UNET(4, 2, 5, 16, 0.1)
model2 = UNET_plusplus(4, 2, 5, 16, 0.1)

TData = []

# Train UNET

cpkt1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
manager1 = tf.train.CheckpointManager(cpkt1, '/content/drive/My Drive/MIC/checkpoints/UNET/', max_to_keep=10, checkpoint_name="L_combined_Unet")

data1 = train(
    dataset, labeled_val_ds, model1, 0.1, 40, loss="combined loss", save_ckpt=True, save_ckpt_name='L_combined_Unet', load_ckpt=False, cpkt=cpkt1, manager=manager1)

TData.append(data1[1:])
print("\n\n\n")

# Train UNET++

cpkt2 = tf.train.Checkpoint(optimizer = tf.keras.optimizers.Adam(), model=model2)
manager2 = tf.train.CheckpointManager(cpkt2, '/content/drive/My Drive/MIC/checkpoints/UNET++/', max_to_keep=10, checkpoint_name="L_combined_Unet++")

data2 = train(
    dataset, labeled_val_ds, model2, 0.1, 40, loss="combined loss", save_ckpt=True, save_ckpt_name='L_combined_Unet++', load_ckpt=False, cpkt=cpkt2, manager=manager2)

TData.append(data2[1:])
print("\n\n\n")


# plotting

coloring = ['blue', 'red']
labels = ['U-NET', 'U-NET++']
metrics = ['training loss', 'validation loss', 'Ones accuracy validation',
           'IOU - validation set', 'validation accuracy']

# Prepare data for plotting
TData = [[row[i] for row in TData] for i in range(len(metrics))]

def plt_data(lossData):
  for k, m in enumerate(metrics):
    for i, data in enumerate(lossData[k]):
      plt.plot(data, label=labels[i], color=coloring[i], marker='o')
    plt.title(metrics[k])     
    plt.legend()    
    # plt.ylim((0, 1))
    plt.show()
 
plt_data(TData)

# Testing

# Visualize test outputs

modelr = UNET(5, 2, 5, 16, 0.1)
cpkt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), model=modelr)

manager = tf.train.CheckpointManager(cpkt, '/content/drive/My Drive/MIC/checkpoints/UNET/', max_to_keep=10, checkpoint_name="L_dice_Unet")
print(manager.latest_checkpoint)
cpkt.restore(manager.latest_checkpoint)

for image in test_ds.take(3):
    print(image.shape)
    y_pred = modelr(image, is_train=False)
    show_batch(image, y_pred, True)

# Evaluate trained model

modelr = UNET(4, 2, 5, 16, 0.1)
cpkt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=tf.keras.optimizers.Adam(), model=modelr)
manager = tf.train.CheckpointManager(cpkt, '/content/drive/My Drive/MIC/checkpoints/UNET++/', max_to_keep=10, checkpoint_name="L_focal_Unet++")
print(manager.latest_checkpoint)
cpkt.restore('/content/drive/My Drive/MIC/checkpoints/UNET++/L_focal_Unet++-1')

loss = 'focal loss'
cce = tf.keras.losses.BinaryCrossentropy()

def val_eval(y_true, y_pred):

    # IoU
    true_ones = tf.reduce_sum(y_true)
    intersection = y_true * y_pred 
    common_ones = tf.reduce_sum(intersection)
    union = y_true + y_pred - intersection 
    total_ones = tf.reduce_sum(union)
    iou = common_ones / total_ones

    total_labels = y_true.shape[0]*y_true.shape[1]*y_pred.shape[2]

    # accuracy - TP + FN / ( all )
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    crct_labels = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
    accuracy = crct_labels / total_labels
    
    # precision - TP / (TP + FP)
    prec = common_ones / tf.reduce_sum(y_pred)

    return iou, accuracy, prec

# Validation        
Val_iou = tf.keras.metrics.Mean()
Val_acc = tf.keras.metrics.Mean()
Val_prec = tf.keras.metrics.Mean()

start_time = time.time()
for x, y_true in labeled_val_ds:   

    y_pred = modelr(x, is_train=False)

    if loss == 'cross entropy':
        val_loss = cce(y_true, y_pred)
    elif loss == 'combined'
        val_loss = 
    elif loss == 'dice loss':
        val_loss = dice_loss(y_true, y_pred)
    else:
        val_loss = focal_loss(y_true, y_pred, 2)            
    Val_loss.update_state(val_loss)
    val_iou, val_acc, val_prec = val_eval(y_true, y_pred)
  
    Val_iou.update_state(val_iou)
    Val_acc.update_state(val_acc)
    Val_prec.update_state(val_prec)

print("\nEval Time : {:.3f}, iou : {:.5f}, accuracy : {:.5f}, precision : {:.5f}".
              format(time.time() - start_time, Val_iou.result(), Val_acc.result(), Val_prec.result()))