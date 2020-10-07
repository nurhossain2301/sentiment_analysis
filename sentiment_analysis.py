#import packages

import numpy as np
import pandas as pd
import glob
from scipy.sparse import csr_matrix
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import re
import string

def imdb_data_preprocess():
    pos = [pd.read_csv(filename, sep='delimiter', header=None, encoding = "ISO-8859-1") for filename in glob.glob("aclImdb/train/pos/*.txt")]
    df_pos = pd.concat(pos, axis=0, ignore_index=True)
    df_pos['row_number'] = df_pos.index + 1
    df_pos['Polarity'] = 1
    df_pos.rename(columns={0: 'text'}, inplace=True)
    df_pos = df_pos[['row_number', 'text', 'Polarity']]
    l = len(df_pos.index)

    neg = [pd.read_csv(filename, sep='delimiter', header=None, encoding = "ISO-8859-1") for filename in glob.glob("aclImdb/train/neg/*.txt")]
    df_neg = pd.concat(neg, axis=0, ignore_index=True)
    df_neg['row_number'] = df_neg.index + 1 + l
    df_neg['Polarity'] = 0
    df_neg.rename(columns={0: 'text'}, inplace=True)
    df_neg = df_neg[['row_number', 'text', 'Polarity']]

    df = pd.concat([df_pos, df_neg], ignore_index = True)
    df = df.sample(frac=1).reset_index(drop=True)
    ll = len(df)

    X_test_neg = [pd.read_csv(filename, sep='delimiter', header=None) for filename in
                  glob.glob("aclImdb/test/neg/*.txt")]
    X_test_neg = pd.concat(X_test_neg, axis=0, ignore_index=True)
    X_test_neg['row_number'] = X_test_neg.index + 1 + ll
    X_test_neg['Polarity'] = 0
    X_test_neg.rename(columns={0: 'text'}, inplace=True)
    X_test_neg = X_test_neg[['row_number', 'text', 'Polarity']]
    lll = len(X_test_neg)

    X_test_pos = [pd.read_csv(filename, sep='delimiter', header=None) for filename in
                  glob.glob("aclImdb/test/pos/*.txt")]
    X_test_pos = pd.concat(X_test_pos, axis=0, ignore_index=True)
    X_test_pos['row_number'] = X_test_pos.index + 1 + ll + lll
    X_test_pos['Polarity'] = 1
    X_test_pos.rename(columns={0: 'text'}, inplace=True)
    X_test_pos = X_test_pos[['row_number', 'text', 'Polarity']]
    X_test = pd.concat([X_test_pos, X_test_neg], ignore_index=True)
    X_full = pd.concat([df, X_test], ignore_index = True)
    X = X_full['text'].to_numpy()
    X_new = X
    for i in range(len(X)):
        X_re = X[i].translate(str.maketrans('', '', string.punctuation))
        X_re = re.sub('<[^<]+?>', '', X_re)
        X_new[i] = X_re
    test_Y = X_test['Polarity'].to_numpy()
    train_Y = df['Polarity'].to_numpy()

    return X_new, test_Y, train_Y, ll

def main():
    X_new, y_test, y_train, l = imdb_data_preprocess()
    #X = X_full['text'].to_numpy()
    stop_words = []
    with open('stopwords.en.txt', 'r') as reader:
        stop_words_en = reader.read().split('\n')


    # unigram CV
    vec_uni = CountVectorizer(strip_accents = 'unicode')
    X_uni = vec_uni.fit_transform(X_new)
    X_train = X_uni[0:l, :]
    X_test = X_uni[l:, :]
    clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=10000, tol=5e-4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print('accuracy:', accuracy)
    f = open('unigram.output.txt', 'w')
    for i in y_pred:
        f.write('%s\n' % i)
    f.close()

    # # bigram CV
    vec_bi = CountVectorizer(ngram_range=(2, 2))
    X_bi = vec_bi.fit_transform(X_new)
    X_train = X_bi[0:l, :]
    X_test = X_bi[l:, :]
    clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=10000, tol=5e-4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print('accuracy:', accuracy)
    f = open('bigram.output.txt', 'w')
    for i in y_pred:
        f.write('%s\n' % i)
    f.close()

    # # unigram Tv
    tvec_uni = TfidfVectorizer(stop_words=stop_words_en)
    tX_uni = tvec_uni.fit_transform(X)
    X_train = tX_uni[0:l, :]
    X_test = tX_uni[l:, :]
    clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=10000, tol=1e-4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print('accuracy:', accuracy)
    f = open('unigramtfidf.output.txt', 'w')
    for i in y_pred:
        f.write('%s\n' % i)
    f.close()

    # # # bigram tv
    tvec_bi = TfidfVectorizer(ngram_range=(2, 2))
    tX_bi = tvec_bi.fit_transform(X_new)
    print(tvec_bi.get_feature_names())
    X_train = tX_bi[0:l, :]
    X_test = tX_bi[l:, :]
    clf = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=10000, tol=3e-4)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print('accuracy:', accuracy)
    f = open('bigramtfidf.output.txt', 'w')
    for i in y_pred:
        f.write('%s\n' % i)
    f.close()


if __name__ == "__main__":
    main()





