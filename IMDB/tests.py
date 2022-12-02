import tensorflow as tf 
import numpy as np


def attention_vect(input_seq):
    result = np.zeros(shape=input_seq.shape)
    for i,pivot_vect in enumerate(input_seq):
      _len = len(input_seq)
      scores = np.zeros(shape=(_len))
      for j,vect in enumerate(input_seq):
        scores[j] = np.dot(pivot_vect,vect.T)
      scores /= np.sqrt(_len)
      scores = tf.nn.softmax(scores).numpy()
      new_pivot_representation = np.zeros(shape=pivot_vect.shape)
      for j,vect in enumerate(input_seq):
        new_pivot_representation += vect * scores[j]  
      result[i] = new_pivot_representation
    return result 
