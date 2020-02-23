# Imports
import sys
import time
import os

import numpy as np
import cv2
import pickle

from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

import tensorflow as tf

import mobilefacenets.TinyMobileFaceNet as TinyMobileFaceNet

from face_detector import FaceDetector

class mobilefacenet(object):
    def __init__(self):
        with tf.Graph().as_default():
            # define placeholder
            self.inputs = tf.compat.v1.placeholder(name='img_inputs',
                                         shape=[None, 112, 112, 3],
                                         dtype=tf.float32)
            self.phase_train_placeholder = tf.compat.v1.placeholder_with_default(tf.constant(False, dtype=tf.bool),
                                                                       shape=None,
                                                                       name='phase_train')
            # identity the input, for inference
            inputs = tf.identity(self.inputs, 'input')

            prelogits, net_points = TinyMobileFaceNet.inference(images=inputs,
                                                                phase_train=self.phase_train_placeholder,
                                                                weight_decay=5e-5)

            self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

            # define sess
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                    gpu_options=gpu_options,
                                    )
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)

            # saver to load pretrained model or save model
            # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
            saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())

            # init all variables
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.sess.run(tf.compat.v1.local_variables_initializer())

            ckpt = tf.train.get_checkpoint_state('./tiny_model')
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_feature(self, inputs):
        inputs = np.expand_dims(inputs, axis=0)
        feed_dict = {self.inputs: inputs, self.phase_train_placeholder: False}
        feature = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return feature
        
facenet = mobilefacenet()
facedetect = FaceDetector('model.pb', gpu_memory_fraction=0.25, visible_device_list='0')

def distance(reference, sample):
  return np.sum(np.square(reference - sample))

# Input is a path to an image, output is the image as a numpy array. Only really necessary for manual debug
def preprocess_image(filename):
    # load image from file
    image_array = cv2.imread(filename)
    # convert to RGB, if needed
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return(image_array)

# Input is an image array, output is a numpy array containing all detected faces from an image 
## Ignores all faces with confidence < 0.99
## Returns a 4D numpy array (n, W, H, C) containing all faces from image, sorted by descending confidence
def extract_faces(image_array, thresh=0.5):
  # detect faces in the image
  boxes, scores = facedetect(image_array, score_threshold=thresh)
  # pull out each bounding box
  faces=np.empty((0,112,112,3))
  boxes = boxes.astype(int)
  for i in boxes:
      y1, x1, y2, x2 = i
      face = image_array[y1:y2, x1:x2]
      face = cv2.resize(face, (112,112))
      face_row = np.expand_dims(face, 0)
      faces = np.append(faces, face_row, 0)
  return faces

# Input is an image array, output is a facenet face embedding
def embed_face(face_pixels):
  embedding = facenet.get_feature(face_pixels)
  return embedding[0]

# Calls embed_face on an image array/label (and the current face/label library), and returns the new face/label library, retrained svm model, and new in/out encoders.
def add_face(face_pixels, label, library=None):
  if library is None:
    face_library = np.empty((0,128))
    label_library = np.empty((0,))
    out_encoder = LabelEncoder()
    out_encoder.classes_ = []
  else:
    face_library = library[0]
    label_library = library[1]
    out_encoder = LabelEncoder()
    out_encoder.classes_ = library[2]

  face = embed_face(face_pixels)
  face_row = np.expand_dims(face, axis=0)
  new_face_library = np.append(face_library, face_row, axis=0)
  new_label_library = np.append(out_encoder.classes_, label)
  # normalise input vectors
  in_encoder = Normalizer(norm='l2')
  new_face_library = in_encoder.transform(new_face_library)
  # label encode targets
  out_encoder.fit(new_label_library)
  new_label_library = out_encoder.transform(new_label_library)
  # fit model
  if len(new_label_library) >= 2:
    svm = SVC(kernel='linear')
    svm.fit(new_face_library, new_label_library)
  else:
    svm = None
  
  new_library = list([new_face_library, new_label_library, out_encoder.classes_, svm])
  return new_library

def remove_face(label, library=None):
  if library is None:
    return
  else:
    face_library = library[0]
    label_library = library[1]
    out_encoder = LabelEncoder()
    out_encoder.classes_ = library[2]
  label = out_encoder.transform([label])
  index = np.where(label_library==label)
  new_face_library = np.delete(face_library, index, axis=0)
  new_label_library = np.delete(out_encoder.classes_, index, axis=0)
  out_encoder.fit(new_label_library)
  new_label_library = out_encoder.transform(new_label_library)
  new_library = list([new_face_library, new_label_library, out_encoder.classes_, svm])
  return new_library

# Calls embed_face on an image array, and uses SVM to return the predicted label
def identify_face(face_pixels, library=None, threshold=0.7):
  if library is None:
    return
  else:
    face_library = library[0]
    label_library = library[1]
    out_encoder = LabelEncoder()
    out_encoder.classes_ = library[2]
    svm = library[3]
  face = embed_face(face_pixels)
  face_row = np.expand_dims(face, axis=0)
  in_encoder = Normalizer(norm='l2')
  face_row = in_encoder.transform(face_row)
  if len(label_library) >= 2:
    prediction = svm.predict(face_row)
  elif len(label_library) == 0:
    prediction = ["Unknown"]
    return prediction
  else:
    prediction = out_encoder.classes_[0]
  ref_index = np.where(label_library == prediction)
  ref_face = face_library[ref_index][0]
  if distance(ref_face, face) < threshold:
    prediction = out_encoder.inverse_transform(prediction)
  else:
    prediction = ["Unknown"]
  return prediction

# Identify and return classifications for ALL faces detected in an image
def identify_all(image_array, library=None, threshold=0.7):
  faces = extract_faces(image_array)
  identities = list()
  for i in faces:
    identification = identify_face(i, library, threshold)
    identities.append(identification)
  return identities

