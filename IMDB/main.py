import tensorflow as tf 
from tensorflow.keras.datasets import imdb
import numpy as np
import tests as ts
from tensorflow import keras 
from tensorflow.keras import layers
import plot as plt

class IMDBClassifier:

  
  def load_dataset(self):
    (self.train_ds,self.train_labels_ds),(self.test_ds,self.test_labels_ds) = imdb.load_data(num_words=self.seq_len)

  def pre_process(self,dataset):
    new_ds = np.zeros((len(dataset),self.seq_len))
    for i in np.arange(len(dataset)):
        for j in dataset[i]:
           new_ds[i,j] = 1
    return new_ds  
  def define_layers(self):
    
    self.model = keras.Sequential([
        layers.Dense(16,activation='relu'),
        layers.Dense(16,activation='relu'),
        layers.Dense(1,activation='sigmoid')
    ])
    


  def uncode_seq(self,seq):
    word_idx = imdb.get_word_index()
    reverse = dict([(value,key) for key,value in word_idx.items()])
    return ' '.join([reverse.get(idx-3,'?') for idx in seq])  

  def fit(self):
    x_val = self.train_ds[:10000]
    x_train = self.train_ds[10000:]
    y_val = self.train_labels_ds[:10000]
    y_train = self.train_labels_ds[10000:]

    history = self.model.fit(x_train,y_train,epochs=3,batch_size=512,
                             validation_data=(x_val,y_val))
    return history                       
  def __init__(self):
    
    
    self.seq_len = 10 * 1000
    self.load_dataset()
    self.train_ds = self.pre_process(self.train_ds)
    self.test_ds = self.pre_process(self.test_ds)
    self.train_labels_ds = np.array(self.train_labels_ds,dtype='float32')
    self.test_labels_ds= np.array(self.test_labels_ds,dtype='float32')
    self.define_layers()
    self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    hist = self.fit()
    results = self.model.evaluate(self.test_ds,self.test_labels_ds)
    print(results)
    plt.plot_validation_loss_history(hist)
   
if __name__ == '__main__':
  IMDBClassifier()