#!/usr/bin/env python
#coding: utf-8
'''
Created on May 1, 2019

@author: xingtong
'''
import os
import nltk
from nltk.tree import Tree
import tnt_tagger_tool
import spacy
import textacy.extract

nlp = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"


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

def token_is_subject_with_action(token):
    nsubj=True #token.dep_ == 'nsubj'
    head_verb=True#token.head.pos=='VERB'
    type=token.ent_type_=='PERSON'
    return nsubj and head_verb and type

if __name__ == '__main__':
#     tnt_tagger=tnt_tagger_tool.get_trained_tnt_tagger()
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,'CCAT')
    ent_list=[]
#     doc_ent_set=set()
    for (file_name, txt) in read_articles(folder_path):
#         print('--------------------------------Begin process %s---------------------------------' % file_name)
        doc=nlp(txt)
#         for sent in doc:
#         for token in doc.ents:
#             if token.label_ in ['ORG']:
#                 cke=chunk_entity()
#                 cke.ent=token.lemma_
#                 cke.verb=token.sent.root.lemma_
#                 ent_list.append(cke) 
                
            
            
        for (ind,sent) in enumerate(doc.sents):
            verb=None
            if sent.root.pos_=='VERB':
                verb=sent.root.lemma_
            else:
                verb='None'
            for ent in sent.ents:
                if ent.label_ in ['ORG']:
                    cke=chunk_entity()
                    cke.ent=ent.lemma_
                    cke.verb=verb
                    cke.sent=sent.lemma_
                    ent_list.append(cke) 
#             #得到一句话的主干
#             st=nlp(sent.string)  
#             subj=''
#             verb=''
#             obj=''
#             for token in st:
#                 if token.dep_ == 'nsubj':
#                     subj=token.lemma_      #Lemmatize
#                 elif token.dep_ == 'ROOT':
#                     verb=token.lemma_
#                 elif token.dep_ == 'pobj':
#                     obj=token.lemma_ 
#             for chunk in sent.noun_chunks:
#                 ents=[ent for ent in chunk.ents if ent.label_ in ['ORG']]
#                 for ent in ents:
#                     cke=chunk_entity()
#                     cke.ent=ent.lemma_
#                     cke.chunk=chunk.string
#                     cke.sent=(sent.string).strip()
#                     cke.sent_subject=subj
#                     cke.sent_verb=verb
#                     cke.sent_object=obj
#                     cke.sent_ind=ind
#                     cke.file_name=file_name
#                     if subj==cke.ent:
#                         cke.type='subj'
#                     elif obj==cke.ent:
#                         cke.type='obj'
#                     ent_list.append(cke)   
#                     doc_ent_set.add(cke.ent)
                    
#         # 提取半结构化语句
#         for ent in doc_ent_set:
#             statements = textacy.extract.semistructured_statements(doc,ent)  
#             print('%s:' % ent)
#             for statement in statements:
#                 subject, verb, fact = statement
#                 print('%s %s %s' %(subject, verb, fact))    
    
    static_dic={}
    for org in ent_list:
        if static_dic.get(org.ent) is None:
            static_dic[org.ent]=[org]
        else:
            static_dic[org.ent].append(org)
    org_list=list(static_dic.items())
#     print(org_list)
    org_list.sort(key=lambda x:len(x[1]), reverse=True)
    print('There are totally %d organizations' % len(org_list))
    for ind,(key,value) in enumerate(org_list):
        print('%d. Orgnization: %s, Count: %d' % (ind+1, key, len(value)))
#         orgs=static_dic.get(org.ent)
        type_dic={}
        act_dic={}
        for org in value:
#             if type_dic.get(org.type) is None:
#                 type_dic[org.type]=[1,{org.verb:1}]
#             else:
#                 type_dic[org.type][0]=+1
#             act_dic=act_dic[org.verb]
            if act_dic.get(org.verb) is None:
                act_dic[org.verb]=1
            else:
                act_dic[org.verb]+=1
                
#             print(org)
#             print('-%s' % (org.sent))
#             print('%s : %s %s' % (org.ent,org.verb,org.type))
#         print('---------------------------------------------------------')
#         type_list=list(type_dic.items())
#         type_list.sort(key=lambda x:x[1][0], reverse=True)
#         for (key,value) in type_list:
#             print('-------------------type: %s, count: %s--------------------' % (key,value[0]))
#         act_list=list(value.items())
        act_list=list(act_dic.items())
        act_list.sort(key=lambda x:x[1], reverse=True)    
        print('act, count')
        for (key,value) in act_list:
            print('%s, %s' % (key,value))
        print()
        print('---------------------------------------------------------')
        if (ind+1)==5:
            break
#         print('------------------------------Process %s finish----------------------------------' % file_name)

    
    
    
    
    
    
    
    