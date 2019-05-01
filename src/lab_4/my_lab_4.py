#!/usr/bin/env python
#coding: utf-8
'''
Created on May 1, 2019

@author: xingtong
'''
import os
import nltk
from nltk.tree import Tree

class chunk_entity(object):
    pass

def read_articles(folder_path):
    '''
    read all articles in a folder
    @param folder: folder path
    @return: a list each item in the list is a article
    '''
    files= os.listdir(folder_path)
    article_list = []
    for file_name in files:
        article=None
        file_path=get_abs_path(folder_path, file_name)
        if not os.path.isdir(file_path):
            with open(file_path, 'rb') as f:
                article=f.read()
            yield file_name,str(article,encoding = "utf-8")

def get_parent_folder_path():
    return os.path.abspath(os.path.dirname(os.getcwd()))

def get_grand_parent_folder_path():
    return os.path.abspath(os.path.join(os.getcwd(), "../.."))

def get_abs_path(base_path,path_names):
    return os.path.join(base_path,path_names)

def get_continuous_chunks(chunks):
    '''
    get continuous chunks
    @param chunks: chunks list, each chunk named entity
    @return: continuous chunks list, 
            each item is a chunk_entity object, chunk_entity_obj.value: a string of continuous chunks;chunk_entity_obj.label:the label of the chunk
            
    '''
    continuous_chunks = []
    current_chunks = []
    for chunk in chunks:
        if type(chunk) == Tree:
            current_chunks.append(chunk)
        elif current_chunks:
            chunk_entity_obj=chunk_entity()
            chunk_entity_obj.value=' '.join([' '.join([token for token, pos in chk.leaves()]) for chk in current_chunks])
            chunk_entity_obj.label=current_chunks[0].label()
            continuous_chunks.append(chunk_entity_obj)
            current_chunks = []

    if current_chunks:
        chunk_entity_obj=chunk_entity()
        chunk_entity_obj.value=' '.join([' '.join([token for token, pos in chk.leaves()]) for chk in current_chunks])
        chunk_entity_obj.label=current_chunks[0].label()
        continuous_chunks.append(chunk_entity_obj)
    return continuous_chunks

if __name__ == '__main__':
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,'data')
    for (file_name, txt) in read_articles(folder_path):
        print('--------------------------------Begin process %s---------------------------------' % file_name)
        for (ind,sent) in enumerate(nltk.sent_tokenize(txt)):
            token_words=nltk.word_tokenize(sent)
            pos_words=nltk.pos_tag(token_words)
            chunks=nltk.ne_chunk(pos_words)
            continuous_chunks=get_continuous_chunks(chunks)
            for chunk_entity_obj in continuous_chunks:
                if chunk_entity_obj.label in ('PERSON','GPE'):
                    print('''%s : %s , in the %dth sentance of the file %s''' % (chunk_entity_obj.value,chunk_entity_obj.label,(ind+1),file_name))
        print('------------------------------Process %s finish----------------------------------' % file_name)
'''
Output:
Chinese : GPE , in the 1th sentance of the file: text1.txt
Wang Yi : PERSON , in the 1th sentance of the file: text1.txt
Trump : PERSON , in the 1th sentance of the file: text1.txt
Beijing : GPE , in the 1th sentance of the file: text1.txt
November : GPE , in the 1th sentance of the file: text1.txt
Mr. Trump : PERSON , in the 2th sentance of the file: text1.txt
Beijing : GPE , in the 2th sentance of the file: text1.txt
China : GPE , in the 2th sentance of the file: text1.txt
Mr. Trump : PERSON , in the 3th sentance of the file: text1.txt
Beijing : GPE , in the 3th sentance of the file: text1.txt
Mr. Pillsbury : PERSON , in the 4th sentance of the file: text1.txt
China : GPE , in the 4th sentance of the file: text1.txt

Person                            True Condition    
                                positive    negative     recall=100%
prodicted outcome    positive    5 (TP)    0 (FP)        precision=100%
                    negative    0 (FN)    10 (TN)
            
Locations                         True Condition    
                                positive    negative     recall=100%
prodicted outcome    positive    5 (TP)    2 (FP)        precision=71%
                     negative    0 (FN)    8 (TN)

'''
    
    
    
    
    
    
    
    