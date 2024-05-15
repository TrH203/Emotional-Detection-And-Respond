# this is emotion predict model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
class model_processing():

    def __init__(self,train_sentence,test_sentence) -> None:
        self.train_sentence = train_sentence
        self.test_sentence = test_sentence
        self.vocab_size = 20000
        self.embedding_dim = 100
        self.max_length = 300
        self.trunc_type='post'
        self.padding_type='post'
        self.oov_tok = "<OOV>"
    
    def tokenizer_def(self):
        self.tokenizer = Tokenizer(oov_token=self.oov_tok)
        self.tokenizer.fit_on_texts(self.train_sentence)
    
    def check_max_sentence(self):
        m = 0
        vt = 0
        for i,v in enumerate(self.train_sentence):
            if m < len(v):
                m = len(v)
                vt = i             
        return (m,self.train_sentence)
    
    
    
    def main_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size,self.embedding_dim,input_length=self.max_length),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(36,activation='relu'),
            tf.keras.layers.Dense(24,activation='relu'),
            tf.keras.layers.Dense(12,activation='relu'),
            tf.keras.layers.Dense(6,activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
    
    def fit_model(self,train_padded,train_label,test_padded,test_label,num_epochs=30):
        history = self.model.fit(train_padded, train_label, epochs=num_epochs, validation_data=(test_padded,test_label), verbose=2)
        
    def predict_model(self,padded):
        return self.model.predict(padded)
    
    def save_model(self,text):
        self.model.save(text)

    def load_model(self,model):
        self.model = model