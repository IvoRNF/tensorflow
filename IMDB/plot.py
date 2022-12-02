import matplotlib.pyplot as plt 



def plot_validation_loss_history(history):
      dict_ = history.history
      loss_values = dict_['loss']
      val_loss_values = dict_['val_loss']
      epochs = range(1,len(loss_values)+1)
      plt.plot(epochs,loss_values,'bo',label='training loss') #blue dot
      plt.plot(epochs,val_loss_values,'b',label='validation loss') #blue line
      plt.title('training and validation loss')
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.legend()
      plt.show() 