"""
classify.py
"""
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold 
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import unittest
import pickle
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

def tokenize(doc, keep_internal_punct=False):
   
    if keep_internal_punct==True:
        
        doc = doc.lower().split()
        i=0
        for word in doc:
            word = (word.strip(string.punctuation))
            doc[i]= word
            i+=1
        i=0
        return np.array(doc)
    else:
        return np.array(re.sub('\W+', ' ', doc.lower()).split())   
    
def token_features(tokens, feats):
  
    for t in tokens:
        if ('token='+ t) not in feats.keys():
            feats['token='+ t] = 1
        else:
            feats['token='+ t] += 1
            
def featurize(tokens, feature_fns):
    
    feats={}
    for feature in feature_fns:
        #print(feature)
        feature(tokens, feats)
    return sorted(feats.items())

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
   
    if vocab == None:
        vocab_feature_dict ={}  #used to create vocab
        row, column, data, feature_list = [], [], [], [] #used for CSR matrix
    
        for t in tokens_list:
            #print(t)
            feature = featurize(t,feature_fns)
            f_dict ={}
            #print(feature)
            for f in feature:
                
                f_dict[f[0]] = f[1]
                if f[0] in vocab_feature_dict:
                    vocab_feature_dict[f[0]] +=1
                else:
                    vocab_feature_dict[f[0]] =1
            feature_list.append(f_dict)
        #print(vocab_feature_dict)
        #Creating vocab
        vocab ={}
        i=0
        for key,value in sorted(vocab_feature_dict.items()):
            if vocab_feature_dict[key] >= min_freq:
                vocab[key] = i
                i += 1
        
        #Creating CSR matrix
        i=0
        for f in feature_list:
            for elem in f:
                if elem in vocab:
                    column.append(vocab[elem])
                    row.append(i)
                    data.append(f[elem])
            i += 1
        #print(vocab)    
        if (len(row)!=0 and len(column)!=0):
            X = csr_matrix((data,(row,column)),shape=(len(tokens_list),len(vocab)))
        else:
            X = csr_matrix(([],([],[])),shape=(0,0))
        return X,vocab
    else:
        
        row, column, data, feature_list = [], [], [], []
    
        for t in tokens_list:
            #print(t)
            feature = featurize(t,feature_fns)
            f_dict ={}
            #print(feature)
            for f in feature:
                
                f_dict[f[0]] = f[1]
            feature_list.append(f_dict)
        
        #print(vocab_feature_dict)
        
        #Creating CSR matrix
        i=0
        for f in feature_list:
            for elem in f:
                if elem in vocab:
                    column.append(vocab[elem])
                    row.append(i)
                    data.append(f[elem])
            i += 1
        #print(vocab)    
        if (len(row)!=0 and len(column)!=0):
            X = csr_matrix((data,(row,column)),shape=(len(tokens_list),len(vocab)))
        else:
            X = csr_matrix(([],([],[])),shape=(0,0))
        return X,vocab

def accuracy_score(truth, predicted):
    
    return len(np.where(truth==predicted)[0]) / len(truth)

def cross_validation_accuracy(clf, X, labels, k):
    
    cv = KFold(len(labels),k)
    accuracies = []
    #print(cv)
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))

    return np.mean(accuracies)

def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    
    combined_feature = []
    for i in range(len(feature_fns)):
        for c in combinations(feature_fns,i+1):
            combined_feature.append(list(c))
    #print(combined_feature)
 
    
    tokens_true = [tokenize(d,keep_internal_punct=True) for d in docs]
    tokens_false = [tokenize(d,keep_internal_punct=False) for d in docs]
    result =[]
    for f in combined_feature:
        for punct in punct_vals:
            for freq in min_freqs:
                if punct == True:
                    X ,vocab= vectorize(tokens_true, f,freq)
                    if (X.nnz != 0):
                        cv = cross_validation_accuracy(LogisticRegression(),X,labels,5)
                        result.append({'features':f ,'punct':punct, 'accuracy':cv,'min_freq':freq})

                else:
                    X,vocab = vectorize(tokens_false, f,freq)
                    if (X.nnz != 0):
                        cv = cross_validation_accuracy(LogisticRegression(),X,labels,5)
                        result.append({'features':f ,'punct':punct, 'accuracy':cv,'min_freq':freq})

    return sorted(result, key=lambda d: -d['accuracy'])

def affin_pos_neg(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


def labels_pred(tok, afinn):
    sorted_tokens = tokenize(tok['text'])
    pos, neg = affin_pos_neg(sorted_tokens, afinn)
    if pos > neg:
        return 1
    elif neg > pos:
         return 0
    else:
        return -1

def make_vocabulary(tokens_list):
    vocabu = defaultdict(lambda: len(vocabu))
    for tokens in tokens_list:
        for token in tokens:
             vocabu[token]
    return vocabu

def fit_best_classifier(docs, labels, best_result):
   
    clf = LogisticRegression()
    X,vocab=vectorize([tokenize(d,best_result["punct"]) for d in docs], best_result['features'], best_result['min_freq'])
    clf.fit(X, labels) 
    return clf,vocab

def parse_test_data(best_result, vocab,tw):
    
    testing_docs =[]
    testing_labels =[]
    testing_docs = [t['text'] for t in tw if 'user' in t]
    #print(docs[1])
    path = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(path.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    testing_labels = np.array([labels_pred(t, afinn) for t in tw])
    #print(len(labels))
    #print(len(docs))
    #testing_docs,testing_labels=read_data(os.path.join('data', 'Test'))
    X_test,vocab=vectorize([tokenize(d,best_result["punct"]) for d in testing_docs], best_result['features'], best_result['min_freq'],vocab=vocab)
    return testing_docs,testing_labels,X_test

def main():
    tweets= pickle.load(open('data.pkl', 'rb'))
    tweets1 = pickle.load(open('test_data.pkl', 'rb'))
    
    docs =[]
    labels =[]
    docs = [t['text'] for t in tweets if 'user' in t]
    #print(docs[1])
    path = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(path.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    labels = np.array([labels_pred(t, afinn) for t in tweets])
    print(len(labels))
    print(len(docs))
        
    feature_fns = [token_features]
    #print(os.path.join('data', 'Train'))
    #docs, labels = read_data(os.path.join('data', 'Train'))
    
    results = eval_all_combinations(docs, labels,
                                    [True,False],
                                    feature_fns,
                                    [2,5,10])
    best_result = results[0]
    #worst_result = results[-1]
    
    tokens = [tokenize(t['text']) for t in tweets]
    vocab = make_vocabulary(tokens)
    print(len(vocab))
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab,tweets1)
    #print(test_docs[0])
    predictions = clf.predict(X_test)
    probablities_label = clf.predict_proba(X_test)

    #index_counter=0
    
    pos_class_counter=0
    neg_class_counter=0
    positive_max_probability=0
    positive_max_probability_counter=0
    negative_max_probability=0
    negative_max_probability_counter=0

    for index in range(len(probablities_label)):
        val=probablities_label[index]
        if np.all(val[1] > positive_max_probability):
            
            positive_max_probability_counter=index
            positive_max_probability=probablities_label[index]
        elif np.all(val[0] > negative_max_probability):
            negative_max_probability_counter=index
            negative_max_probability=probablities_label[index]
    #print(len(predictions))
    for index in range(len(predictions)):
        if predictions[index] == 1:
            pos_class_counter+=1
        elif predictions[index] == 0:
            neg_class_counter+=1
            
    f5 = open('classify_data.txt', 'w+')
    f5.write("Number of instances per class found: "+"\n")
    f5.write("Number of instances of Positive class found: "+str(pos_class_counter)+"\n")
    f5.write("Number of instances of Negative class found: "+str(neg_class_counter)+"\n\n")

    f5.write("One example from each class: "+"\n")        
    
    #print(test_docs[positive_max_probability_counter])
    f5.write("Example of Positive Class: "+"\n")
    #for i in range(0,10):
        #print(test_docs[i])
    f5.write(test_docs[positive_max_probability_counter]+"\n")
    
    #print("Negative Document")        
    #print(test_docs[negative_max_probability_counter])
    f5.write("Example of Negative Class: "+"\n")
    #for i in range(0,10):
        #print(test_docs[-i])
    f5.write(test_docs[negative_max_probability_counter].decode('utf-8').encode('cp850','replace').decode('cp850')+"\n")
    
    f5.close()
    
if __name__ == '__main__':
    main()
