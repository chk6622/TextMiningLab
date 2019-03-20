#!/usr/bin/env python
#coding: utf-8
'''
Created on Mar 19, 2019

@author: xingtong
'''
import nltk
from nltk.corpus import brown
from pickle import dump,load
import string
import operator
from functools import reduce

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
    @return: train_tagged_sents,test_tagged_sents
    '''
    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_sents = brown.sents(categories='news')
    size = int(len(brown_tagged_sents) * training_percentage)
    train_tagged_sents = brown_tagged_sents[:size]
    test_tagged_sents = brown_tagged_sents[size:]
    return train_tagged_sents,test_tagged_sents,brown_sents[size:]

#get all tagger
unigram_tagger=get_tagger(filename.get('unigram'))
tnt_tagger=get_tagger(filename.get('tnt'))
perceptron_tagger=get_tagger(filename.get('perceptron'))
crf_tagger=nltk.tag.CRFTagger()
try:
    crf_tagger.set_model_file(filename.get('crf'))
except Exception:
    crf_tagger=None

train_tagged_sents=None
test_tagged_sents=None

#get the training and the test data
train_tagged_sents,test_tagged_sents,brown_sents=get_dataset()

#init all tagger
if not unigram_tagger:
    print('Begin init unigram tagger')
    #Instantiate,training Unigram tagger
    unigram_tagger = nltk.UnigramTagger(train_tagged_sents)
    #store unigram_tagger
    store_tagger(unigram_tagger,filename.get('unigram'))
    print('Init unigram tagger finish')


if not tnt_tagger:
    print('Begin init Tnt tagger')
    #Instantiate,training TnT tagger
    tnt_tagger = nltk.tag.tnt.TnT()
    tnt_tagger.train(train_tagged_sents)
    #store tnt_tagger
    store_tagger(tnt_tagger,filename.get('tnt'))
    print('Init Tnt tagger finish')
    
if not perceptron_tagger:
    print('Begin init perceptron tagger')
    #Instantiate,training Perceptron tagger
    perceptron_tagger=nltk.tag.perceptron.PerceptronTagger()
    perceptron_tagger.train(train_tagged_sents)
    #store perceptron_tagger
    store_tagger(perceptron_tagger, filename.get('perceptron'))
    print('Init perceptron tagger finish')

if not crf_tagger:
    print('Begin init crf tagger')
    #Instantiate,training,store CRF tagger
    crf_tagger=nltk.tag.CRFTagger()
    crf_tagger.train(train_tagged_sents,filename.get('crf'))
    print('Init crf tagger finish')


f1_list=[]
unigram_tagger_sents=unigram_tagger.tag_sents(brown_sents)
tag_set_1_d=[x[1] for item in unigram_tagger_sents for x in item]
fd = nltk.FreqDist(tag_set_1_d)
print('----------------unigram_tagger------------------')
fd.tabulate(10)
f1_unigram=unigram_tagger.evaluate(test_tagged_sents)
f1_list.append(('unigram_tagger',f1_unigram))

tnt_tagger_sents=tnt_tagger.tag_sents(brown_sents)
tag_set_1_d=[x[1] for item in tnt_tagger_sents for x in item]
fd = nltk.FreqDist(tag_set_1_d)
print('----------------tnt_tagger------------------')
fd.tabulate(10)
f1_tnt=tnt_tagger.evaluate(test_tagged_sents)
f1_list.append(('tnt_tagger',f1_tnt))

perceptron_tagger_sents=perceptron_tagger.tag_sents(brown_sents)
tag_set_1_d=[x[1] for item in perceptron_tagger_sents for x in item]
fd = nltk.FreqDist(tag_set_1_d)
print('----------------perceptron_tagger------------------')
fd.tabulate(10)
f1_perceptron=perceptron_tagger.evaluate(test_tagged_sents)
f1_list.append(('perceptron_tagger',f1_perceptron))
 
crf_tagger_sents=crf_tagger.tag_sents(brown_sents)
tag_set_1_d=[x[1] for item in crf_tagger_sents for x in item]
fd = nltk.FreqDist(tag_set_1_d)
print('----------------crf_tagger------------------')
fd.tabulate(10)
f1_crf=crf_tagger.evaluate(test_tagged_sents)
f1_list.append(('crf_tagger',f1_crf))

f1_list.sort(key=(lambda x:x[1]), reverse=True)
for item in f1_list:
    print('--------F1 value--------')
    print('%s : %f' % (item[0],item[1]))
print('The best one is %s' % f1_list[0][0])


