#!/usr/bin/env python
#coding: utf-8
'''
Created on Mar 19, 2019

@author: xingtong
'''
import nltk
from nltk.corpus import brown
from pickle import dump,load

filename={}
filename['unigram']='unigram_tagger.pkl'
filename['tnt']='tnt_tagger.pkl'
filename['perceptron']='perceptron_tagger.pkl'
filename['crf']='crf_tagger.pkl'

def store_tagger(tagger,tagger_name):
    '''
    Store a trained model
    @param tagger:the model which will be stored 
    @param tagger_name: model file name 
    '''
    output = open(tagger_name, 'wb')
    dump(tagger, output, -1)
    output.close()
    
def get_tagger(tagger_name):
    '''
    get model file
    @param tagger_name:file name
    @return: tagger 
    '''
    tagger=None
    input=None
    try:
        input = open(tagger_name, 'rb')
        tagger = load(input)
    except Exception:
        pass
    if input:
        input.close()
    return tagger

def get_dataset(training_percentage=0.8):
    '''
    get data set
    @param training_percentage: the percentage of training set account for total data
    @return: train_sents,test_sents
    '''
    brown_tagged_sents = brown.tagged_sents(categories='news', tagset='universal')
    size = int(len(brown_tagged_sents) * training_percentage)
    train_sents = brown_tagged_sents[:size]
    test_sents = brown_tagged_sents[size:]
    return train_sents,test_sents

#get all tagger
unigram_tagger=get_tagger(filename.get('unigram'))
tnt_tagger=get_tagger(filename.get('tnt'))
perceptron_tagger=get_tagger(filename.get('perceptron'))
crf_tagger=nltk.tag.CRFTagger()
try:
    crf_tagger.set_model_file(filename.get('crf'))
except Exception:
    crf_tagger=None

train_sents=None
test_sents=None

#get the training and the test data
train_sents,test_sents=get_dataset()

#init all tagger
if not unigram_tagger:
    print('Begin init unigram tagger')
    #Instantiate,training Unigram tagger
    unigram_tagger = nltk.UnigramTagger(train_sents)
    #store unigram_tagger
    store_tagger(unigram_tagger,filename.get('unigram'))
    print('Init unigram tagger finish')


if not tnt_tagger:
    print('Begin init Tnt tagger')
    #Instantiate,training TnT tagger
    tnt_tagger = nltk.tag.tnt.TnT()
    tnt_tagger.train(train_sents)
    #store tnt_tagger
    store_tagger(tnt_tagger,filename.get('tnt'))
    print('Init Tnt tagger finish')
    
if not perceptron_tagger:
    print('Begin init perceptron tagger')
    #Instantiate,training Perceptron tagger
    perceptron_tagger=nltk.tag.perceptron.PerceptronTagger()
    perceptron_tagger.train(train_sents)
    #store perceptron_tagger
    store_tagger(perceptron_tagger, filename.get('perceptron'))
    print('Init perceptron tagger finish')

if not crf_tagger:
    print('Begin init crf tagger')
    #Instantiate,training,store CRF tagger
    crf_tagger=nltk.tag.CRFTagger()
    crf_tagger.train(train_sents,filename.get('crf'))
    print('Init crf tagger finish')

tags=unigram_tagger.tag_sents(test_sents)
print(len(tags))
# tmp=[a for a,b in tags[0]]
# fd = nltk.FreqDist([a+'' for a,b in tmp])
# fd.tabulate()


