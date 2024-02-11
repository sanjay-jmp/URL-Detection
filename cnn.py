import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import random
import os
import tldextract

import tensorflow as tf
from tensorflow.python.util import deprecation
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers, backend, metrics
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

pd.set_option('max_colwidth', 50)
pio.templates.default = "presentation"
pd.options.plotting.backend = "plotly"
deprecation._PRINT_DEPRECATION_WARNINGS = False

data = pd.read_csv('predata.csv')
print(data.head())
val_size = 0.2
train_data, val_data = train_test_split(data, test_size=val_size, stratify=data['label'], random_state=0)
print(data.label.value_counts())

def parsed_url(url):
    # Extract subdomain, domain, and domain suffix from url
    result = tldextract.extract(url)
    subdomain = '' if result.subdomain == '' else result.subdomain
    domain = '' if result.domain == '' else result.domain
    domain_suffix = '' if result.suffix == '' else result.suffix
    return [subdomain, domain, domain_suffix]

def extract_url(data):
    extract_url_data = [parsed_url(url) for url in data['url']]
    extract_url_data = pd.DataFrame(extract_url_data, columns=['subdomain', 'domain', 'domain_suffix'])
    data = data.reset_index(drop=True)
    data = pd.concat([data, extract_url_data], axis=1)
    return data

data = extract_url(data)
train_data = extract_url(train_data)
val_data = extract_url(val_data)


print('Unique Domains :',data.domain.nunique())
print('Unique Subdomains :',data.subdomain.nunique())
print('Unique Domain Suffixes: ',data.domain_suffix.nunique())

tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)
tokenizer.fit_on_texts(train_data['url'])
n_char = len(tokenizer.word_index.keys())

train_seq = tokenizer.texts_to_sequences(train_data['url'])
val_seq = tokenizer.texts_to_sequences(val_data['url'])

print(train_seq[0])



sequence_length = np.array([len(i) for i in train_seq])
sequence_length = np.percentile(sequence_length, 99).astype(int)

train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)

print(f'{train_seq[0]}')

unique_value = {}
for feature in ['subdomain', 'domain', 'domain_suffix']:
    label_index = {label: index for index, label in enumerate(train_data[feature].unique())}
    label_index[''] = list(label_index.values())[-1] + 1
    unique_value[feature] = label_index['']
    train_data.loc[:, feature] = [label_index[val] if val in label_index else label_index[''] for val in
                                  train_data.loc[:, feature]]
    val_data.loc[:, feature] = [label_index[val] if val in label_index else label_index[''] for val in
                                val_data.loc[:, feature]]

for data in [train_data, val_data]:
    data.loc[:, 'label'] = [0 if i == 'good' else 1 for i in data.loc[:, 'label']]

# print(train_data.head())

import pickle

# Save label index to a file
with open('label_index.pkl', 'wb') as label_index_file:
    pickle.dump(label_index, label_index_file)

# Save sequence length to a file
with open('sequence_length.pkl', 'wb') as sequence_length_file:
    pickle.dump(sequence_length, sequence_length_file)

# Save unique value to a file
with open('unique_value.pkl', 'wb') as unique_value_file:
    pickle.dump(unique_value, unique_value_file)



def convolution_block(x):
    conv_3_layer = layers.Conv1D(64, 3, padding='same', activation='elu')(x)
    conv_5_layer = layers.Conv1D(64, 5, padding='same', activation='elu')(x)
    conv_layer = layers.concatenate([x, conv_3_layer, conv_5_layer])
    conv_layer = layers.Flatten()(conv_layer)
    return conv_layer

def embedding_block(unique_value, size, name):
    input_layer = layers.Input(shape=(1,), name=name + '_input')
    embedding_layer = layers.Embedding(unique_value, size, input_length=1)(input_layer)
    return input_layer, embedding_layer


def create_model(sequence_length, n_char, unique_value):
    input_layer = []

    sequence_input_layer = layers.Input(shape=(sequence_length,), name='url_input')
    input_layer.append(sequence_input_layer)

    char_embedding = layers.Embedding(n_char + 1, 32, input_length=sequence_length)(sequence_input_layer)
    conv_layer = convolution_block(char_embedding)

    entity_embedding = []
    for key, n in unique_value.items():
        size = 4
        input_l, embedding_l = embedding_block(n + 1, size, key)
        embedding_l = layers.Reshape(target_shape=(size,))(embedding_l)
        input_layer.append(input_l)
        entity_embedding.append(embedding_l)

    fc_layer = layers.concatenate([conv_layer, *entity_embedding])
    fc_layer = layers.Dropout(rate=0.5)(fc_layer)

    fc_layer = layers.Dense(128, activation='elu')(fc_layer)
    fc_layer = layers.Dropout(rate=0.2)(fc_layer)

    output_layer = layers.Dense(1, activation='sigmoid')(fc_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy',
                  metrics=[metrics.Precision(), metrics.Recall()])
    return model


backend.clear_session()
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)
#
# model = create_model(sequence_length, n_char, unique_value)
#
#
# train_x = [train_seq, train_data['subdomain'], train_data['domain'], train_data['domain_suffix']]
# train_y = train_data['label'].values
# train_seq_tensor = tf.convert_to_tensor(train_seq, dtype=tf.int32)
# train_subdomain_tensor = tf.convert_to_tensor(train_data['subdomain'].values, dtype=tf.int32)
# train_domain_tensor = tf.convert_to_tensor(train_data['domain'].values, dtype=tf.int32)
# train_domain_suffix_tensor = tf.convert_to_tensor(train_data['domain_suffix'].values, dtype=tf.int32)
# train_y_tensor = tf.convert_to_tensor(train_y, dtype=tf.float32)
#
# # Define the input as a list of tensors
# train_x = [train_seq_tensor, train_subdomain_tensor, train_domain_tensor, train_domain_suffix_tensor]
#
#
# early_stopping = [EarlyStopping(monitor='val_precision', patience=5, restore_best_weights=True, mode='max')]
# history = model.fit(train_x, train_y_tensor, batch_size=1024, epochs=5, verbose=1, validation_split=0.2, shuffle=True, callbacks=early_stopping)
# model.save('model.h5')
# print(model.summary())


model=tf.keras.models.load_model('model.h5')
val_x = [val_seq, val_data['subdomain'], val_data['domain'], val_data['domain_suffix']]
val_y = val_data['label'].values
val_seq_tensor = tf.convert_to_tensor(val_seq, dtype=tf.int32)
val_subdomain_tensor = tf.convert_to_tensor(val_data['subdomain'].values, dtype=tf.int32)
val_domain_tensor = tf.convert_to_tensor(val_data['domain'].values, dtype=tf.int32)
val_domain_suffix_tensor = tf.convert_to_tensor(val_data['domain_suffix'].values, dtype=tf.int32)
val_y = tf.convert_to_tensor(val_y, dtype=tf.float32)
val_x = [val_seq_tensor, val_subdomain_tensor, val_domain_tensor, val_domain_suffix_tensor]
val_pred = model.predict(val_x)
val_pred_h = np.where(val_pred[:, 0] >= 0.5, 1, 0)

print(f'Validation Data:\n{val_data.label.value_counts()}')
print(f'\n\nConfusion Matrix:\n{confusion_matrix(val_y, val_pred_h)}')
print(f'\n\nClassification Report:\n{classification_report(val_y, val_pred_h)}')

fpr, tpr, thresholds = roc_curve(val_y, val_pred)
roc_auc=auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()