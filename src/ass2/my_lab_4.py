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
from collections import Counter

nlp = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"

class root_dep_static_tool(object):
    def __init__(self,entity_name):
        self.entity_name=entity_name
        self.root_dep_count=[]  #(root_dep,count,(actions)) item1 (action,count,(sentence)) item2
#         self.actions=[]  #(action,count,(sentence)) item2
    
    def static(self,entity):
        if self.entity_name!=entity.get_ent():
            return
        flag=False
        for item1 in self.root_dep_count:  
            if entity.root_dep==item1[0]:
                flag=True
                item1[1]+=1
                action=entity.verb
                isExist=False
                #['compound', 2, [None, 1, [Ethiopian Airlines Boeing 737 flight in fatal crash]]]
                for item2 in item1[2]:
#                     print(item2)
                    if item2 is not None and action==item2[0]:
                        isExist=True
                        item2[1]+=1
                        item2[2].append(entity.sent)
#                         item2[2]+=entity.children
                if not isExist:
                    item1[2].append([action,1,[entity.sent]])
#                     item1[2].append([action,1,entity.children])
        if not flag:
            self.root_dep_count.append([entity.root_dep,1,[[entity.verb,1,[entity.sent]]]])
#             self.root_dep_count.append([entity.root_dep,1,[[entity.verb,1,entity.children]]])
            
        

class static_tool(object):
    def __init__(self):
        self.entities_count={}
#         for entity_name in entities_name_list:
#             rdst=root_dep_static_tool(entity_name)
#             self.entities_count[rdst.entity_name]=[rdst,1]
        
        
    def add_entity(self,entity):
        ec=self.entities_count.get(entity.get_ent())
        if ec is not None:
            ec[1]+=1
            ec[0].static(entity)
        else:
            rdst=root_dep_static_tool(entity.get_ent())
            rdst.static(entity)
            self.entities_count[rdst.entity_name]=[rdst,1]
            
    def output_static_results(self):
        items=list(self.entities_count.items())
        items.sort(key=lambda x:x[1][1], reverse=True)
        for (ind,item) in enumerate(items):
            if ind==5:
                break
            print('\n%d. Orgnization: %s, Count: %d' % (ind+1,item[0],item[1][1]))
            root_deps=item[1][0].root_dep_count
            root_deps.sort(key=lambda x:x[1],reverse=True)
            for root_dep in root_deps:
                rd=root_dep[0]
                rd_count=root_dep[1]
                if not (('obj' in rd) or ('sub' in rd)):
                    continue 
                print('\n-------------------root dependent: %s, count: %d--------------------' % (rd,rd_count))
                actions=root_dep[2]
                actions.sort(key=lambda x:x[1],reverse=True)
                for action in actions:
                    if action[0] is None:
                        continue
                    print('%s : %d' % (action[0],action[1]))
                    sents=action[2]
                    for (ind,sent) in enumerate(sents):
                        print('(%d). %s' % (ind+1,sent))
                    
                
        

class chunk_entity(object):
    def __init__(self):
        self.last_iob=None
        self.ent=[]
#         self.ent_type=None
        self.verb=None
        self.ent_complete=False
        self.sent=None
        self.root_dep=None
        self.adj_list=[]
        self.children=None
        
    def get_ent(self):
        return ' '.join(self.ent)
        
    def complete_entity(self,token=None):
        self.ent_complete=False
        if token is not None:
            ent_type=token.ent_type_
            iob=token.ent_iob_
            ent_text=token.lemma_
            last_iob=None
            root_dep=token.dep_
            children=[child for child in token.children]
            
            #add element into the entity 
            if ent_type == 'ORG' and iob in ['I','O','B']:
                if (iob=='B' and self.last_iob in [None,'O']) or (iob=='I' and self.last_iob in ['B','I']):           
                    last_iob=iob
                    self.ent.append(ent_text)
                    self.sent=self.get_sentence_trunk(token.sent)
                    self.root_dep=root_dep
                    self.children=children
                    if self.verb is None:
                        head=token.head.lemma_
                        hp=token.head.pos_
                        if hp in ['VERB']:
                            self.verb=head
            #conduct whether the entity is completed
            cond_1=self.last_iob in ['B','I']
            cond_2=iob in ['O','B']
            self.ent_complete=all([cond_1,cond_2])
            if last_iob is not None: 
                self.last_iob=last_iob
        return self.ent_complete
    
    def get_sentence_trunk(self,sentence):
        n_subject=''
        np_subject=''
        object=''
        d_object=''
        verb=''
        i_object=''
        p_object=''
        sent=[]
        for word in sentence:
            if 'nsubj' in word.dep_:
                n_subject=word.lemma_
            elif 'npsubj' in word.dep_:
                np_subject=word.lemma_
            elif 'obj' in word.dep_:
                object=word.lemma_
            elif 'dobj' in word.dep_:
                d_object=word.lemma_
            elif 'iobj' in word.dep_:
                i_object = word.lemma_
            elif 'pobj' in word.dep_:
                p_object = word.lemma_
            elif 'ROOT' in word.dep_:
                verb = word.lemma_
        sent.append(n_subject)
        sent.append(np_subject)
        sent.append(verb)
        sent.append(object)
        sent.append(d_object)
        sent.append(i_object)
        sent.append(p_object)
        return ' '.join(sent) 
        

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

# def token_is_subject_with_action(token):
#     nsubj=True #token.dep_ == 'nsubj'
#     head_verb=True#token.head.pos=='VERB'
#     type=token.ent_type_=='PERSON'
#     return nsubj and head_verb and type

if __name__ == '__main__':
    folder_path='CCAT'
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,folder_path)
    ent_list=[]
    for (file_name, txt) in read_articles(folder_path):
        doc=nlp(txt)
        for sent in doc.sents:
            cke=chunk_entity()
            for token in sent:              
                if cke.complete_entity(token=token):   
                    ent_list.append(cke)
                    cke=chunk_entity()
                    cke.complete_entity(token=token)
                    
    static_tool=static_tool()     
    for ent in ent_list: 
        static_tool.add_entity(ent)
    static_tool.output_static_results()
#     static_dic={}
#     for org in ent_list:
#         if not org.ent_complete:
#             continue
#         ent=org.get_ent()
#         if static_dic.get(ent) is None:
#             static_dic[ent]=[org]
#         else:
#             static_dic[ent].append(org)
#     org_list=list(static_dic.items())
#     org_list.sort(key=lambda x:len(x[1]), reverse=True)
#     print('There are totally %d organizations' % len(org_list))
#     for ind,(key,value) in enumerate(org_list):
#         print('%d. Orgnization: %s, Count: %d' % (ind+1, key, len(value)))
#         root_dep_dic={}
#         act_dic={}
#         for org in value:
#             if root_dep_dic.get(org.root_dep) is None:
#                 root_dep_dic[org.root_dep]=[1,{org.verb:1}]
#             else:
#                 root_dep_dic[org.root_dep][0]+=1
#                 
#                 act_dic=root_dep_dic[org.root_dep][1]
#                 if act_dic.get(org.verb) is None:
#                     act_dic[org.verb]=1
#                 else:
#                     act_dic[org.verb]+=1
#                     
#         root_dep_list=list(root_dep_dic.items())
#         root_dep_list.sort(key=lambda x:x[1][0], reverse=True)
#         for (key,value) in root_dep_list:
#             print('-------------------type: %s, count: %s--------------------' % (key,value[0]))
#             act_list=list(value[1].items())
#             act_list.sort(key=lambda x:x[1], reverse=True)    
#             print('act, count')
#             for (key,value) in act_list:
#                 print('%s, %s' % (key,value))
#             print()
#             print('---------------------------------------------------------')
#         if (ind+1)==1:
#             break
#     print('------------------------------Process %s finish----------------------------------' % file_name)

    
    
    
    
    
    
    
    