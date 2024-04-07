

import sys
import os

sys.path.append("/storage/research/data/abusive_content_2/Abusive_Content_Detection")




import sys
import os

import shutil

try:
    from dotenv import find_dotenv, load_dotenv
except:
    pass

import argparse

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "/storage/research/data/abusive_content_2/Abusive_Content_Detection"))
except:
    sys.path.append(os.path.join(os.getcwd(), "/storage/research/data/abusive_content_2/Abusive_Content_Detection"))
    
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
except:
    sys.path.append(os.path.join(os.getcwd(), '../'))
    
import pandas as pd
import numpy as np

import pickle
from collections import Counter
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from sklearn.preprocessing import LabelEncoder

try:
    import wandb
    load_dotenv(find_dotenv())
    wandb.login(key=os.environ['WANDB_API_KEY'])
    from wandb.keras import WandbCallback
    _has_wandb = True
except:
    _has_wandb = False

import tokenizers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig, BertTokenizer

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src import data, models

pd.options.display.max_colwidth = -1


# In[ ]:


print (_has_wandb)


# In[ ]:


parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')


parser.add_argument('--train_data', type=str, default="/storage/research/data/abusive_content_2/Abusive_Content_Detection", required=False,
                    help='train data')

parser.add_argument('--val_data', type=str, default=None, required=False,
                    help='validation data')
parser.add_argument('--test_data', type=str, default=None, required=False,
                    help='test data')

parser.add_argument('--transformer_model_pretrained_path', type=str, default='roberta-base', required=False,
                    help='transformer model pretrained path or huggingface model name')
parser.add_argument('--transformer_config_path', type=str, default='roberta-base', required=False,
                    help='transformer config file path or huggingface model name')
parser.add_argument('--transformer_tokenizer_path', type=str, default='roberta-base', required=False,
                    help='transformer tokenizer file path or huggingface model name')

parser.add_argument('--max_text_len', type=int, default=50, required=False,
                    help='maximum length of text')
parser.add_argument('--max_char_len', type=int, default=200, required=False,
                    help='maximum length of text')
parser.add_argument('--max_word_char_len', type=int, default=20, required=False,
                    help='maximum length of text')

parser.add_argument('--emb_dim', type=int, default=128, required=False,
                    help='maximum length of text')
parser.add_argument('--n_layers', type=int, default=2, required=False,
                    help='maximum length of text')
parser.add_argument('--n_units', type=int, default=128, required=False,
                    help='maximum length of text')

parser.add_argument('--epochs', type=int, default=500, required=False,
                    help='number of epochs')
parser.add_argument('--lr', type=float, default=.001, required=False,
                    help='learning rate')
parser.add_argument('--early_stopping_rounds', type=int, default=50, required=False,
                    help='number of epochs for early stopping')
parser.add_argument('--lr_schedule_round', type=int, default=30, required=False,
                    help='number of epochs for learning rate scheduling')

parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                    help='eval batch size')



parser.add_argument('--model_save_path', type=str, default="/storage/research/data/abusive_content_2/Abusive_Content_Detection", required=False,
                    help='seed')

parser.add_argument('--wandb_logging', type=bool, default=True, required=False,
                    help='wandb logging needed')

parser.add_argument('--seed', type=int, default=42, required=False,
                    help='seed')


args, _ = parser.parse_known_args()


# In[ ]:


tf.random.set_seed(args.seed)
np.random.seed(args.seed)


# In[ ]:


df = pd.read_csv("/storage/research/data/abusive_content_2/Abusive_Content_Detection/hate_speech.tsv", \
                 sep='\t',header=None,usecols=[0,1])
df.columns = ['text','category']
df = df.dropna()
df = df[df.text != '']

kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
for train_index, test_index in kf.split(df.text):
    break

df['type'] = 'hate'

hate_train_df = df.iloc[train_index]
kf2 = KFold(n_splits=2, shuffle=True, random_state=args.seed)
for val_index, test_index in kf2.split(df.iloc[test_index].text):
    break

hate_val_df = df.iloc[val_index]
hate_test_df = df.iloc[test_index]




# In[ ]:


train_df = pd.concat([hate_train_df,hate_train_df],axis=0)
val_df = pd.concat([hate_val_df,hate_val_df],axis=0)
test_df = pd.concat([hate_test_df,hate_test_df],axis=0)

print (train_df.shape, val_df.shape, test_df.shape)


# In[ ]:


train_df.head(5)


# In[ ]:


train_df.text = train_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
val_df.text = val_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
test_df.text = test_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))

train_df = train_df[train_df.text != '']
val_df = val_df[val_df.text != '']
test_df = test_df[test_df.text != '']


# In[ ]:


hate_train_df.text = hate_train_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
hate_val_df.text = hate_val_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))
hate_test_df.text = hate_test_df.text.apply(lambda x: data.preprocessing.clean_tweets(x))

hate_train_df = hate_train_df[hate_train_df.text != '']
hate_val_df = hate_val_df[hate_val_df.text != '']
hate_test_df = hate_test_df[hate_test_df.text != '']


# In[ ]:


train_df.text.apply(lambda x: len(x)).describe()


# In[ ]:


train_df.text.apply(lambda x: len(x.split())).describe()


# In[ ]:


model_save_dir = args.model_save_path

try:
    os.makedirs(model_save_dir)
except OSError:
    pass


# In[ ]:


hate_train_df.category.value_counts()


# In[ ]:


train_df = train_df[train_df.category.str.contains('yes|no')]
val_df = val_df[val_df.category.str.contains('yes|no')]
test_df = test_df[test_df.category.str.contains('yes|no')]


# In[ ]:


hate_train_df = hate_train_df[hate_train_df.category.str.contains('yes|no')]
hate_val_df = hate_val_df[hate_val_df.category.str.contains('yes|no')]
hate_test_df = hate_test_df[hate_test_df.category.str.contains('yes|no')]


# In[ ]:


model_save_dir


# In[ ]:


hate_train_df.category, label2idx = data.data_utils.convert_categorical_label_to_int(hate_train_df.category.values, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))

hate_val_df.category, _ = data.data_utils.convert_categorical_label_to_int(hate_val_df.category.values, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))

hate_test_df.category, _ = data.data_utils.convert_categorical_label_to_int(hate_test_df.category.values, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))


# In[ ]:


train_df.head(5)


# In[ ]:


print (label2idx)


# In[ ]:


idx2label = {i:w for (w,i) in label2idx.items()}
le = LabelEncoder()
train_df['category']=le.fit_transform(train_df['category'])
val_df['category']=le.fit_transform(val_df['category'])
test_df['category']=le.fit_transform(test_df['category'])

# ### Learn tokenizer

# In[ ]:


data.custom_tokenizers.custom_wp_tokenizer(hate_train_df.text.values, args.model_save_path, args.model_save_path)
tokenizer = BertTokenizer.from_pretrained(args.model_save_path)


# In[ ]:


word_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=50000, split=' ',oov_token=1)
char_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, split='',oov_token=1)

word_tokenizer.fit_on_texts(hate_train_df.text.values)
char_tokenizer.fit_on_texts(hate_train_df.text.values)


# In[ ]:


transformer_train_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(train_df, 'text', tokenizer, args.max_char_len)

word_train_inputs = word_tokenizer.texts_to_sequences(train_df.text.values)
word_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_train_inputs, maxlen=args.max_text_len)

subword_train_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(train_df.text.values)])

char_train_inputs = char_tokenizer.texts_to_sequences(train_df.text.values)
char_train_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_train_inputs, maxlen=args.max_char_len)

train_outputs = data.data_utils.compute_output_arrays(train_df, 'category')

transformer_val_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(hate_val_df, 'text', tokenizer, args.max_char_len)

word_val_inputs = word_tokenizer.texts_to_sequences(hate_val_df.text.values)
word_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_val_inputs, maxlen=args.max_text_len)

subword_val_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(hate_val_df.text.values)])

char_val_inputs = char_tokenizer.texts_to_sequences(hate_val_df.text.values)
char_val_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_val_inputs, maxlen=args.max_char_len)

val_outputs = data.data_utils.compute_output_arrays(hate_val_df, 'category')

transformer_test_inputs, _, _ = data.data_utils.compute_transformer_input_arrays(test_df, 'text', tokenizer, args.max_char_len)

word_test_inputs = word_tokenizer.texts_to_sequences(test_df.text.values)
word_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(word_test_inputs, maxlen=args.max_text_len)

subword_test_inputs = np.asarray([data.data_utils.subword_tokenization(text, char_tokenizer, args.max_text_len, args.max_word_char_len) \
                        for text in tqdm(test_df.text.values)])

char_test_inputs = char_tokenizer.texts_to_sequences(test_df.text.values)
char_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(char_test_inputs, maxlen=args.max_char_len)

test_outputs = data.data_utils.compute_output_arrays(test_df, 'category')

train_outputs = tf.keras.utils.to_categorical(train_outputs, \
                                                    num_classes=train_df.category.nunique())
val_outputs = tf.keras.utils.to_categorical(val_outputs, \
                                                    num_classes=train_df.category.nunique())
test_outputs = tf.keras.utils.to_categorical(test_outputs, \
                                                    num_classes=train_df.category.nunique())

print (transformer_train_inputs.shape, subword_train_inputs.shape, word_train_inputs.shape, char_train_inputs.shape, \
       train_outputs.shape)
print (transformer_val_inputs.shape, subword_val_inputs.shape, word_val_inputs.shape, char_val_inputs.shape, \
       val_outputs.shape)
print (transformer_test_inputs.shape, subword_test_inputs.shape, word_test_inputs.shape, char_test_inputs.shape, \
       test_outputs.shape)


# ### Modeling

# In[ ]:

train_texts = tokenizer.batch_decode(transformer_train_inputs, skip_special_tokens=True)
val_texts = tokenizer.batch_decode(transformer_val_inputs, skip_special_tokens=True)
test_texts = tokenizer.batch_decode(transformer_test_inputs, skip_special_tokens=True)
# Compute TF-IDF features using tokenized text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()
val_tfidf = tfidf_vectorizer.transform(val_texts).toarray()# Check the shapes of TF-IDF matrices
test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()# Check the shapes of TF-IDF matrices


print("Shape of train_tfidf:", train_tfidf.shape)
print("Shape of val_tfidf:", val_tfidf.shape)


n_words = len(word_tokenizer.word_index)+1
n_chars = len(char_tokenizer.word_index)+1
n_subwords = tokenizer.vocab_size
tfidf_shape = train_tfidf.shape[1]
n_out = 2


# In[ ]:


from src.models.models import *

#all_models = {HAN.__name__: HAN, \
#              CMSA.__name__: CMSA, CS_ELMO_without_words.__name__: CS_ELMO_without_words}

all_models = {HAN.__name__: HAN}


# In[ ]:


_has_wandb = False


# In[ ]:

print("Shapes of Input Data:")
print("Transformer Input Shape:", transformer_train_inputs.shape)
print("Subword Input Shape:", subword_train_inputs.shape)
print("Word Input Shape:", word_train_inputs.shape)
print("Char Input Shape:", char_train_inputs.shape)
print("TF-IDF Input Shape:", train_tfidf.shape)
print("Output Shape:", train_outputs.shape)


if os.path.exists(os.path.join(args.model_save_path,'results.csv')):
  results = pd.read_csv(os.path.join(args.model_save_path,'results.csv'))
  index = results.shape[0]
  print (results)
else:
  results = pd.DataFrame(columns=['config','weighted_f1','macro_f1'])
  index = 0

for model_name, model_ in all_models.items():
    
    for loss in ['ce','focal']:
        
        for use_features in [True, False]:
            
            if use_features == False:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=n_out,max_word_char_len=args.max_word_char_len,\
                                             max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                             n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
            else:
                model = model_(word_vocab_size=n_words,char_vocab_size=n_chars,wpe_vocab_size=n_subwords, n_out=n_out,vectorizer_shape=tfidf_shape, max_word_char_len=args.max_word_char_len,\
                                             max_text_len=args.max_text_len, max_char_len=args.max_char_len,\
                                             n_layers=args.n_layers, n_units=args.n_units, emb_dim=args.emb_dim)
            
            if use_features == True:
                print ("Running {} with features for {} loss".format(model_name, loss))
            else:
                print ("Running {} without features for {} loss".format(model_name, loss))

            print (model.summary())

            if loss == 'focal':
                model.compile(loss=models.utils.categorical_focal_loss(alpha=1), optimizer='adam', metrics=['accuracy', models.utils.f1_keras]) #binary_crossentropy
            elif loss == 'ce':
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', models.utils.f1_keras]) 

            lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, \
                                                  patience=args.lr_schedule_round, verbose=1, mode='auto', min_lr=0.000001)
            config = {
                  'text_max_len': args.max_text_len,
                  'char_max_len': args.max_char_len,
                  'word_char_max_len': args.max_word_char_len,
                  'n_units': args.n_units,
                  'emb_dim': args.emb_dim,
                  'n_layers': args.n_layers,
                  'epochs': args.epochs,
                  "learning_rate": args.lr,
                  "model_name": model_name,
                  "loss": loss,
                  "use_features": use_features
                }

            if use_features == True:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_with_features.h5'.format(model_name, config['loss']))
            else:
                model_save_path = os.path.join(args.model_save_path, '{}_{}_without_features.h5'.format(model_name, config['loss']))

            f1callback = models.utils.F1Callback(model, [word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf],\
                                      val_outputs, \
                                      filename=model_save_path, \
                                      patience=args.early_stopping_rounds)

            K.clear_session()

            if _has_wandb and args.wandb_logging:
                wandb.init(project='hate_speech_detection',config=config)
                model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs, \
                      validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs), \
                          epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback, WandbCallback()], verbose=2)
            else:
                model.fit([word_train_inputs, char_train_inputs, subword_train_inputs, transformer_train_inputs, train_tfidf], train_outputs, \
                      validation_data=([word_val_inputs, char_val_inputs, subword_val_inputs, transformer_val_inputs, val_tfidf], val_outputs), \
                          epochs=args.epochs,batch_size=args.train_batch_size, callbacks=[lr, f1callback], verbose=2)


            model.load_weights(model_save_path)

            test_pred = model.predict([word_test_inputs, char_test_inputs, subword_test_inputs, transformer_test_inputs, test_tfidf])

            report = classification_report([idx2label[i] for i in test_outputs.argmax(-1)], \
                                           [idx2label[i] for i in test_pred.argmax(-1)])

            f1 = f1_score([idx2label[i] for i in test_outputs.argmax(-1)], \
                                           [idx2label[i] for i in test_pred.argmax(-1)], average='weighted')

            print (report, f1)
            
            results.loc[index,'config'] = str(config)
            results.loc[index, 'weighted_f1'] = f1_score([idx2label[i] for i in test_outputs.argmax(-1)], \
                                           [idx2label[i] for i in test_pred.argmax(-1)], average='weighted')
            results.loc[index, 'macro_f1'] = f1_score([idx2label[i] for i in test_outputs.argmax(-1)], \
                                           [idx2label[i] for i in test_pred.argmax(-1)], average='macro')
            
            index += 1
            
            results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)


# In[ ]:


results


# In[ ]:


results.to_csv(os.path.join(args.model_save_path,'results.csv'),index=False)


# In[ ]:




