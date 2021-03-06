#Import dependencies
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Activation, Dropout, Dense,CuDNNLSTM, Embedding,GRU, CuDNNGRU
from keras.callbacks import *
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
import pandas as pd
import sys

#Get data from google drive
def get_from_drive():
  from google.colab import drive
  drive.mount('/content/drive')
  with open('/content/drive/My Drive/Colab Notebooks/entexna.txt', 'r') as f: 
    text = f.read()
  return text

#Get file from text
def get_from_git():
  #get raw link of data on github
  url='https://raw.githubusercontent.com/orestislampridis/Greek-Lyric-Generation/master/char_level_lstm/entexna.txt?token=ANTGNDJE42Q36BYI7IFYRZS6G4TE6'
  path_to_file = tf.keras.utils.get_file('shakespeare.txt', url)
  text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
  return text

text=get_from_git()

def check_text(t):
  print('the first 100 characters are:',repr(text[:100]))  #read the first 100 characters of doc
  n=len(t)
  print ('Length of text: %i characters' %n) #lenght=number of characters in text
  v=sorted(set(t)) #making the vocabulary of characters
  n_v=len(v) 
  print('number of unique characters: %i' %n)
  return n,v,n_v

n_char,vocab,n_vocab=check_text(text)

char2int=dict((i, c) for c, i in enumerate(vocab)) #map characters to int
int2char=dict((i, c) for i, c in enumerate(vocab)) #map int to char (for "translation")

#print(char2int) #print the result of mapping the characters in the vocabulary

print('\nMapping text...')
text_as_int=np.array([char2int[c] for c in text]) #map the data as int
# Show a sample of our data mapped from text to integers
print ('%s --[mapped to] -- > %s' %(repr(text[100:119]), text_as_int[100:119]))

print('\nMaking samples(sequences) and deviding data to input and target...')
seq_length = 100 #how many characters per sequence
#i.e seq_length=3 text=καλή, input=καλ, target=ή
target=[]
input=[]
step=5 #this step determines how many sequences we want
for i in range (0,n_char-seq_length,step):

  input.append(text_as_int[i:i+seq_length]) 
  target.append(text_as_int[i+seq_length])

print('Input and target data example:')
print("input 2:", "".join([int2char[c] for c in input[2]]))
print("target 2:", int2char[target[2]])


n_samples=len(input)
print("\nNumber of samples:",n_samples)

print('\nReshaping data to feed RNN...')
#We can use the reshape() function on the NumPy array to reshape this one-dimensional array into a two dimensional array 
inputR=np.reshape(input,(n_samples, seq_length))
print("The input representation of: ", "".join([int2char[c] for c in input[0][:13]]),"is now:")
print(inputR[0][:13])
#We represent the target values with One Hot Encoding.
targetE= np_utils.to_categorical(target)
print("The target representation of: ",int2char[target[60]]," is now:\n",targetE[60])
print("/The shape of the input data is:",inputR.shape)
print("The shape of the target data is:",targetE.shape)

print('\nBuilding model...')
model= Sequential()
rnn_size=512
#embedding layer
model.add(Embedding(n_samples, seq_length,input_length=seq_length, trainable=True))
#input layer
model.add(Bidirectional( CuDNNLSTM(rnn_size, return_sequences=True)))
#Hidden layers 
model.add(Bidirectional( CuDNNLSTM(rnn_size)))
#Dropout layer(avoid overfitting)
model.add(Dropout(0.2))
#Output layer
model.add(Dense(targetE.shape[1]))
#Activation function
model.add(Activation('softmax'))
adam = Adam(lr=0.001)
#compile model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
#model details
model.summary()

print('\nCreating callbacks..')

filepath="/content/drive/My Drive/Colab Notebooks/CheckpointsLyricsGen/epochs:{epoch:03d}-val_acc:{val_acc:.5f}.hdf5"
#Folder called CheckpointsLyricsGen in drive
#each file will be stored with epoch number and validation accuracy
#these files contain weights of your neural network

print('Callbacks created at:',filepath[:63])

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose = 1, save_best_only = False, mode ='max')
#the arguments passed in the above code it is monitoring validation accuracy 

callbacks_list = [checkpoint]
#a list so that you can append any other callbacks to this list and pass it in fit function while training 
#all the methods in the list will be called after every epoch

#if we need to train more: uncomment the code below with the correct checkpoint 

model.load_weights('/content/drive/My Drive/Colab Notebooks/CheckpointsLyricsGen/epochs:015-val_acc:0.47429.hdf5')

print('\nTraining model...')

#fit the model
model.fit(inputR,
          targetE,
          epochs=50,
          batch_size=128,
          shuffle= True,
          initial_epoch=16,
          callbacks=callbacks_list,
          validation_split = 0.2,
          validation_data = None,
          validation_steps = None)

#Load weights for generation
                                                                                     #choose the right filename!!!
model.load_weights('/content/drive/My Drive/Colab Notebooks/CheckpointsLyricsGen/epochs:005-val_acc:0.50984.hdf5')                                                                                    
#compile model                                                                       
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

print('\nGenerating text...')

def random_seed():
  start = np.random.randint(0, len(input)-1)
  random_pattern = input[start]
  print('Seed : ')
  print("\"",''.join([int2char[v] for v in random_pattern]), "\"\n")
  return random_pattern

def set_seed():
  seed="Θάλασσα"
  seed_int=([char2int[c] for c in seed])
  pad_len=seq_length-len(seed_int)   
  set_pattern=np.pad(seed_int,(pad_len,0),constant_values=char2int[" "]) #we need to pad the seed so it can be the correct shape
  return set_pattern



pattern = random_seed()   #Choose what type of seed we want



# How many characters you want to generate
generated_characters = 300

results=[]

for i in range(generated_characters):
    x = np.reshape(pattern, ( 1, len(pattern)))
    
    prediction = model.predict(x,verbose = 0)
        
    index = np.argmax(prediction)

    result = int2char[index]

    results.append(result)
        
    pattern = np.append(pattern,index)
    
    pattern = pattern[1:len(pattern)]
print("Generated text:")
print("\"",''.join(results), "\"\n")    
print('\nDone')
