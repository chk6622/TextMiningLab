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
import os



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
    @return: train_tagged_sents,test_tagged_sents, test_brown_sents
    '''
    brown_tagged_sents = brown.tagged_sents(categories='news')
    brown_sents = brown.sents(categories='news')
    size = int(len(brown_tagged_sents) * training_percentage)
    train_tagged_sents = brown_tagged_sents[:size]
    test_tagged_sents = brown_tagged_sents[size:]
    test_brown_sents = brown_sents[size:]
    return train_tagged_sents,test_tagged_sents,test_brown_sents

def read_articles(folder_path):
    '''
    read all articles in a folder
    @param folder: folder path
    @return: a list each item in the list is a article
    '''
    files= os.listdir(folder_path) 
    rootdir = os.getcwd()
    article_list = []
    for file in files:
        article=None
        if not os.path.isdir(file):
            file_path=os.path.join(rootdir, folder_path, file)
            with open(file_path, 'rb') as f:
                article=f.read()
            article_list.append(str(article,encoding = "utf-8"))
    return article_list

if __name__=='__main__':
    filename={}
    filename['unigram']='unigram_tagger.pkl'
    filename['tnt']='tnt_tagger.pkl'
    filename['perceptron']='perceptron_tagger.pkl'
    filename['crf']='crf_tagger.pkl'
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
    train_tagged_sents,test_tagged_sents,test_brown_sents=get_dataset()
    
    #to train all tagger
    if not unigram_tagger:
        print('Begin to train unigram tagger')
        #Instantiate,training Unigram tagger
        unigram_tagger = nltk.UnigramTagger(train_tagged_sents)
        #store unigram_tagger
        store_tagger(unigram_tagger,filename.get('unigram'))
        print('training unigram tagger finished')
    
    
    if not tnt_tagger:
        print('Begin to train Tnt tagger')
        #Instantiate,training TnT tagger
        tnt_tagger = nltk.tag.tnt.TnT()
        tnt_tagger.train(train_tagged_sents)
        #store tnt_tagger
        store_tagger(tnt_tagger,filename.get('tnt'))
        print('Training Tnt tagger finished')
        
    if not perceptron_tagger:
        print('Begin to train perceptron tagger')
        #Instantiate,training Perceptron tagger
        perceptron_tagger=nltk.tag.perceptron.PerceptronTagger()
        perceptron_tagger.train(train_tagged_sents)
        #store perceptron_tagger
        store_tagger(perceptron_tagger, filename.get('perceptron'))
        print('Training perceptron tagger finished')
    
    if not crf_tagger:
        print('Begin to train crf tagger')
        #Instantiate,training,store CRF tagger
        crf_tagger=nltk.tag.CRFTagger()
        crf_tagger.train(train_tagged_sents,filename.get('crf'))
        print('Training crf tagger finished')
    
    #tabulate+evaluate
    f1_list=[]
    unigram_tagger_sents=unigram_tagger.tag_sents(test_brown_sents)
    tag_set_1_d=[x[1] for item in unigram_tagger_sents for x in item]
    fd = nltk.FreqDist(tag_set_1_d)
    print('----------------unigram_tagger------------------')
    fd.tabulate(10)
    f1_unigram=unigram_tagger.evaluate(test_tagged_sents)
    f1_list.append(('unigram_tagger',f1_unigram))
    print()
    
    tnt_tagger_sents=tnt_tagger.tag_sents(test_brown_sents)
    tag_set_1_d=[x[1] for item in tnt_tagger_sents for x in item]
    fd = nltk.FreqDist(tag_set_1_d)
    print('----------------tnt_tagger------------------')
    fd.tabulate(10)
    f1_tnt=tnt_tagger.evaluate(test_tagged_sents)
    f1_list.append(('tnt_tagger',f1_tnt))
    print()
    
    perceptron_tagger_sents=perceptron_tagger.tag_sents(test_brown_sents)
    tag_set_1_d=[x[1] for item in perceptron_tagger_sents for x in item]
    fd = nltk.FreqDist(tag_set_1_d)
    print('----------------perceptron_tagger------------------')
    fd.tabulate(10)
    f1_perceptron=perceptron_tagger.evaluate(test_tagged_sents)
    f1_list.append(('perceptron_tagger',f1_perceptron))
    print()
     
    crf_tagger_sents=crf_tagger.tag_sents(test_brown_sents)
    tag_set_1_d=[x[1] for item in crf_tagger_sents for x in item]
    fd = nltk.FreqDist(tag_set_1_d)
    print('----------------crf_tagger------------------')
    fd.tabulate(10)
    f1_crf=crf_tagger.evaluate(test_tagged_sents)
    f1_list.append(('crf_tagger',f1_crf))
    print()
    
    f1_list.sort(key=(lambda x:x[1]), reverse=True)
    for item in f1_list:
        print('--------F1 value--------')
        print('%s : %f' % (item[0],item[1]))
    print('The best one is %s\n' % f1_list[0][0])
    
    #article key words processing
    folder_path='test_doc'
    article_list=read_articles(folder_path)
    
    tagged_text_list=[]
    for article in article_list:
        tokens = nltk.word_tokenize(article)
    #     print(len(tokens))
        text = nltk.Text(tokens)
        tagged_sents=crf_tagger.tag(text)
        tagged_text_list+=tagged_sents
        
    key_words_str='airline,flight,boeing,737,CEO,pilot'
    key_word_list=key_words_str.lower().split(',')
    key_word_precent={}
    all_nouns=[word.lower() for word,tag in tagged_text_list if 'N' in tag]
    
    # all_tag=set([tag for word,tag in tagged_text_list if 'N' in tag])
    # print(all_tag)
    
    total_word_num=len(all_nouns)
    for key_word in key_word_list:
        same_word_num=0
        for word in all_nouns:
            if key_word in word:
                same_word_num+=1
    #     present=same_word_num/total_word_num
        key_word_precent[key_word]=same_word_num
    
    print('|%10s|%25s|%20s|%21s|' % ('Key words','The number of key words','Total nouns', 'Percentage'))
    print('---------------------------------------------------------------------------')
    
    for item in sorted(key_word_precent.items(),key=(lambda x:x[1]),reverse=True):
        print('|%10s|%25d|%20d|%20.3f%%|' % (item[0],item[1],total_word_num,(item[1]/total_word_num)*100))



'''
Executing results:

----------------unigram_tagger------------------
None   IN   NN   AT    ,    .   JJ  NNS   NP   CC 
2559 2080 1994 1836 1159  963  785  772  656  599 

----------------tnt_tagger------------------
 Unk   IN   NN   AT    ,    .  NNS   JJ   NP   CC 
2559 2224 2033 1831 1158  958  774  740  643  596 

----------------perceptron_tagger------------------
  NN   IN   AT    ,  NNS   JJ    .   NP   CC   RB 
2646 2125 1676 1159 1146 1109  963  933  601  577 

----------------crf_tagger------------------
  NN   IN   AT   NP  NNS    ,    .   JJ   CC  VBD 
2709 2231 1835 1259 1187 1159  963  941  592  553 

--------F1 value--------
crf_tagger : 0.911460
--------F1 value--------
perceptron_tagger : 0.886266
--------F1 value--------
tnt_tagger : 0.832747
--------F1 value--------
unigram_tagger : 0.802688
The best one is crf_tagger

| Key words|  The number of key words|         Total nouns|           Percentage|
---------------------------------------------------------------------------------
|   airline|                      119|                3923|               3.033%|
|    flight|                       62|                3923|               1.580%|
|     pilot|                       33|                3923|               0.841%|
|    boeing|                       22|                3923|               0.561%|
|       ceo|                       15|                3923|               0.382%|
|       737|                        8|                3923|               0.204%|
'''














