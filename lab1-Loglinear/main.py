# our target is to build a log linear model for the data
# and use it to predict the class of a piece of news
# 1. read the data
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
data_path = './ag_news_csv'
train_file = os.path.join(data_path, 'train.csv')
test_file = os.path.join(data_path, 'test.csv')
columns = ['class_index', 'title', 'description']
train_data = pd.read_csv(train_file, header=None, names=columns)
test_data = pd.read_csv(test_file, header=None, names=columns)
print('load data done')

# 2. extract the feature.
# use the title and description as the feature
import re
train_data['text'] = train_data['title'] + ' ' + train_data['description']
test_data['text'] = test_data['title'] + ' ' + test_data['description']
# use the class_index as the target
train_data['class_index'] = train_data['class_index'] - 1
test_data['class_index'] = test_data['class_index'] - 1
# embed the text
# count the frequency of each word in four distinct classes
word_freq = [{} for _ in range(4)]
for index, row in train_data.iterrows():
    # filter out the non-alphabet characters
    text = row['text']
    clear_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    clear_text = clear_text.lower()
    words = clear_text.split()
    for word in words:
        if word not in word_freq[row['class_index']]:
            word_freq[row['class_index']][word] = 1
        else:
            word_freq[row['class_index']][word] += 1
# sort the words by frequency in each class
sorted_word_freq = [sorted(freq.items(), key=lambda x: x[1], reverse=True) for freq in word_freq]
k = 200
# select the top k words in each class
selected_words = [[word for word, freq in freq_list[:k]] for freq_list in sorted_word_freq]
selected_words = [{word: i for i, word in enumerate(words)} for words in selected_words]
vocab_sizes = [len(words) for words in selected_words]
vocab_size = sum(vocab_sizes)
# build the feature
X_train = []
X_test = []
for index, row in tqdm(train_data.iterrows()):
    text = row['text']
    clear_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    clear_text = clear_text.lower()
    words = clear_text.split()
    feature = [0] * vocab_size
    for word in words:
        for i in range(4):
            if word in selected_words[i]:
                feature[i * vocab_sizes[i] + selected_words[i][word]] += 1
    X_train.append(feature)
for index, row in tqdm(test_data.iterrows()):
    text = row['text']
    clear_text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    clear_text = clear_text.lower()
    words = clear_text.split()
    feature = [0] * vocab_size
    for word in words:
        for i in range(4):
            if word in selected_words[i]:
                feature[i * vocab_sizes[i] + selected_words[i][word]] += 1
    X_test.append(feature)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = train_data['class_index'].values
y_test = test_data['class_index'].values
print('extract feature done')

# 3. build the model without using any library
# we will use the gradient descent to train the model
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
# initialize the weight
n_class = 4
learning_rate = 0.001
n_epoch = 10
n_feature = X_train.shape[1]
W = np.random.rand(n_class, n_feature)
# shuffle the train data
index = np.arange(X_train.shape[0])
np.random.shuffle(index)
X_train = X_train[index]
y_train = y_train[index]
losses = []
# train the model
print('\n==================start training==================')
for epoch in range(n_epoch):
    loss = 0
    correct = 0
    progress_bar = tqdm(range(X_train.shape[0]), desc='epoch: %d' % epoch)
    if epoch > 5:
        learning_rate = 0.0001
    for i in progress_bar:
        x = X_train[i].reshape(-1, 1)
        y = np.zeros((n_class, 1))
        y[y_train[i]] = 1
        # forward
        z = W.dot(x)
        a = softmax(z)
        losses.append(-np.log(a[y_train[i]][0]))
        loss += -np.log(a[y_train[i]][0])
        # backward
        dz = a - y
        dW = dz.dot(x.T)
        W -= learning_rate * dW
        if np.argmax(a) == np.argmax(y):
            correct += 1
        progress_bar.set_postfix({'loss': loss / (i + 1), 'accuracy': correct / (i + 1)})
    print('epoch:', epoch, 'loss:', loss / X_train.shape[0])
    print('train accuracy:', correct / X_train.shape[0])
# save the model
np.save('W.npy', W)

# 4. evaluate the model
print('\n==================start predicting==================')
# implement accuracy and f1 score without using any library
def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)
# calculate the f1 score without using any library
def f1_score(y_true, y_pred):
    TP = [0, 0, 0, 0]
    FP = [0, 0, 0, 0]
    FN = [0, 0, 0, 0]
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            TP[y_true[i]] += 1
        else:
            FP[y_pred[i]] += 1
            FN[y_true[i]] += 1
    precision = [0, 0, 0, 0]
    recall = [0, 0, 0, 0]
    f1 = [0, 0, 0, 0]
    for i in range(4):
        precision[i] = TP[i] / (TP[i] + FP[i])
        recall[i] = TP[i] / (TP[i] + FN[i])
        # if both precision and recall are 0, f1 will be 0
        if precision[i] + recall[i] == 0:
            f1[i] = 0
        else:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    f1 = np.mean(f1)
    return f1

# predict the train data
y_pred = []
for i in tqdm(range(X_train.shape[0])):
    x = X_train[i].reshape(-1, 1)
    z = W.dot(x)
    a = softmax(z)
    y_pred.append(np.argmax(a))
y_pred = np.array(y_pred)
print('train accuracy:', accuracy(y_train, y_pred))
print('train f1 score:', f1_score(y_train, y_pred))
# predict the test data
y_pred = []
for i in tqdm(range(X_test.shape[0])):
    x = X_test[i].reshape(-1, 1)
    z = W.dot(x)
    a = softmax(z)
    y_pred.append(np.argmax(a))
y_pred = np.array(y_pred)
print('test accuracy:', accuracy(y_test, y_pred))
print('test f1 score:', f1_score(y_test, y_pred))