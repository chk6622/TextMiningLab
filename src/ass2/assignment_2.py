#!/usr/bin/env python
#coding: utf-8
'''
Created on May 1, 2019

@author: xingtong
'''
import os
# import nltk
# from nltk.tree import Tree
# import tnt_tagger_tool
import spacy
import neuralcoref
import subject_object_extraction
from textacy.extract import subject_verb_object_triples
# import textacy.extract
# from collections import Counter

nlp_1 = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"
neuralcoref.add_to_pipe(nlp_1)  #add NeuralCoref to the pipline of spacy to solve coreference resolution
nlp_2 = spacy.load('en_core_web_sm')

# class root_dep_static_tool(object):
#     def __init__(self,entity_name):
#         self.entity_name=entity_name
#         self.root_dep_count=[]  #(root_dep,count,(actions)) item1 (action,count,(sentence)) item2
# #         self.actions=[]  #(action,count,(sentence)) item2
#     
#     def static(self,entity):
#         if self.entity_name!=entity.get_ent():
#             return
#         flag=False
#         for item1 in self.root_dep_count:  
#             if entity.root_dep==item1[0]:
#                 flag=True
#                 item1[1]+=1
#                 action=entity.verb
#                 isExist=False
#                 #['compound', 2, [None, 1, [Ethiopian Airlines Boeing 737 flight in fatal crash]]]
#                 for item2 in item1[2]:
# #                     print(item2)
#                     if item2 is not None and action==item2[0]:
#                         isExist=True
#                         item2[1]+=1
#                         item2[2].append(entity.sent)
# #                         item2[2]+=entity.children
#                 if not isExist:
#                     item1[2].append([action,1,[entity.sent]])
# #                     item1[2].append([action,1,entity.children])
#         if not flag:
#             self.root_dep_count.append([entity.root_dep,1,[[entity.verb,1,[entity.sent]]]])
# #             self.root_dep_count.append([entity.root_dep,1,[[entity.verb,1,entity.children]]])
            
        

class static_tool(object):
    def __init__(self):
        self.entities_count={}
        
        
    def add_entity(self,entity):
        ec=self.entities_count.get(entity.get_ent()) #{ent_name:[ent]}
        if ec is not None:
            ec.append(entity)
        else:
            self.entities_count[entity.get_ent()]=[entity]
            
    def get_sentence_trunk(self,sentence):
        return [' '.join((str(sub),str(verb),str(obj))) for (sub,verb,obj) in subject_verb_object_triples(sentence)]
            
    def output_static_results(self):
        items=list(self.entities_count.items())
        items.sort(key=lambda x:len(x[1]), reverse=True)
        for (ind,item) in enumerate(items):
            if ind==5:
                break
            ent_name=item[0]
            ent_list=item[1]
            
#             ents=item[1]
            distinct_ents=list(set([(ent.sent,ent.file_name) for ent in ent_list]))
            file_list=[(trunk,file_name) for (sent,file_name) in distinct_ents for trunk in self.get_sentence_trunk(sent)]
            
            file_dic={}
            for file in file_list:
                if file_dic.get(file[1]) is None:
                    file_dic[file[1]]=[file[0]]
                else:
                    file_dic[file[1]].append(file[0])
                    
            file_name_list=list(file_dic.keys())      
            print('\n%d.  %s is found %d times in %d files.' % (ind+1,ent_name,len(ent_list),len(file_name_list)))
#             print('%s is found in %d files' % (ent_name,len(file_name_list)))
            file_name_list.sort(reverse=False)
            for file_name in file_name_list:
                trunk_list=file_dic.get(file_name)
                print('File name: %s' % file_name)
                for (ind,trunk) in enumerate(trunk_list):
                    print(' (%d) %s' % (ind+1,trunk))
                    
                
        

class chunk_entity(object):
    def __init__(self,file_name=None):
        self.last_iob=None
        self.ent=[]
        self.file_name=file_name
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

            if ent_type == 'ORG' and iob in ['I','O','B']:
                if (iob=='B' and self.last_iob in [None,'O']) or (iob=='I' and self.last_iob in ['B','I']):           
                    last_iob=iob
                    self.ent.append(ent_text)
                    self.sent=token.sent
            #conduct whether the entity is completed
            cond_1=self.last_iob in ['B','I']
            cond_2=iob in ['O','B']
            self.ent_complete=all([cond_1,cond_2])
            if last_iob is not None: 
                self.last_iob=last_iob
        return self.ent_complete
    
    

        

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


if __name__ == '__main__':
    folder_path='CCAT'
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,folder_path)
    nlp_1.remove_pipe("ner")  #remove ner module from pipe for speeding up
    ent_list=[]
    for (file_name, txt) in read_articles(folder_path):
        doc=nlp_1(txt)
        doc=nlp_2(doc._.coref_resolved)  #coreference resolution
        for sent in doc.sents:
            cke=chunk_entity(file_name=file_name)
            for token in sent:              
                if cke.complete_entity(token=token):   
                    ent_list.append(cke)
                    cke=chunk_entity(file_name=file_name)
                    cke.complete_entity(token=token)
                    
    static_tool=static_tool()     
    for ent in ent_list: 
        static_tool.add_entity(ent)
    static_tool.output_static_results()

    
    
    
    
    
    
    
    