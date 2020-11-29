from keras import backend as K
from keras import losses
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Bidirectional, LSTM,Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import pickle
from keras.preprocessing.text import Tokenizer
import numpy as np

VOCAB_SIZE = 44
DROPOUT_RATE =  0.5
EMBEDDING_SIZE = 45
LEARNING_RATE = 10e-7
NUMBER_EPOCHS = 10


X = np.array([[[1], [2], [3], [4], [5.]]])
y = np.array([[1.]])
essay_size = X.shape[1]
print essay_size
            
essay = Input(shape=(5, 1), dtype='int32', name='essay')
embedding_layer = Embedding(output_dim=EMBEDDING_SIZE, input_dim=VOCAB_SIZE+1, input_length=5 , name='embedding_layer', trainable=False)
essay_embedded = embedding_layer(essay)

#first_lstm_layer = Bidirectional(LSTM(10,return_sequences=False, name='first_lstm'), merge_mode='concat')
#temp_out_1 = first_lstm_layer(essay_embedded)
#dropout_layer = Dropout(DROPOUT_RATE,name='first_dropout_layer')
#first_lstm_out = dropout_layer(temp_out_1)

#second_lstm_layer = Bidirectional(LSTM(10, name='first_lstm'), merge_mode='concat')
#temp_out_2 = second_lstm_layer(first_lstm_out)
#dropout_layer = Dropout(DROPOUT_RATE,name='second_dropout_layer')
#second_lstm_out = dropout_layer(temp_out_2)
flat = Flatten()(essay_embedded)
dense_layer = Dense(1, name='output_layer')
out = dense_layer(flat)

model = Model(inputs=essay, outputs=out)
model.summary()

#rmsprop = RMSprop(lr=0.1, rho=0.9, epsilon=None, decay=0.0)
train_checkpoint_file = '/home/nam/Desktop/AspiringMinds/lstm_datafiles/lstm_weights/train_weights.{epoch:02d}--{loss:.2f}.hdf5'
train_checkpoint = ModelCheckpoint(train_checkpoint_file, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
model.compile(optimizer = 'sgd', loss='mse', metrics=['accuracy'])
model.fit(X, y, epochs=1, verbose=1, batch_size=1,callbacks=[train_checkpoint])
model.reset_states()