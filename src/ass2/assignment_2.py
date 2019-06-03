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
from gensim.summarization import summarize
# import textacy.extract
# from collections import Counter

nlp_1 = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"
neuralcoref.add_to_pipe(nlp_1)  #add NeuralCoref to the pipline of spacy to solve coreference resolution
nlp_2 = spacy.load('en_core_web_sm')
            
        

class static_tool(object):
    def __init__(self):
        self.entities_count={}
        self.output_result=[]
        
    def output_static_results_file(self,file_path):
        if not os.path.isdir(file_path):
            with open(file_path, 'w') as f:
                for line in self.output_result:
                    f.write(line+'\n')
        
    def add_entity(self,entity):
        ec=self.entities_count.get(entity.get_ent()) #{ent_name:[ent]}
        if ec is not None:
            ec.append(entity)
        else:
            self.entities_count[entity.get_ent()]=[entity]
            
    def get_sentence_trunk(self,sentence):
        '''
        get all the subj+verb+obj phrase from the sentence 
        @param sentence:
        @return: a list, every item in the list is a subj+verb+obj phrase
        '''
        return [' '.join((str(sub),str(verb),str(obj))) for (sub,verb,obj) in subject_verb_object_triples(sentence)]
    
    def get_ent_from_doc(self,doc,ent_text):
        ent_return=None
        for ent in doc:
            if ent.lemma_==ent_text:
                ent_return=ent
                break
        return ent_return
            
    def output_static_results(self):
        items=list(self.entities_count.items())
        items.sort(key=lambda x:len(x[1]), reverse=True)
        
        
        for (ind,item) in enumerate(items):
            file_summary_dic={}
            if ind==5:
                break
            ent_name=item[0]
            ent_list=item[1]
            
            distinct_ents=list(set([(ent.sent,ent.file_name,ent.sent_index,ent.file_path) for ent in ent_list])) #for two entities which are the same name in the one sentence, we think they are one entity
            trunk_list=[(trunk,file_name,sent_index,file_path) for (sent,file_name,sent_index,file_path) in distinct_ents for trunk in self.get_sentence_trunk(sent)] 
#             trunk_list=[('',file_name,sent_index,file_path) for (sent,file_name,sent_index,file_path) in distinct_ents]
            
            file_dic={}
            for trunk in trunk_list:  #get the trunks of all sentence which contains the entity words in one file
                if file_dic.get(trunk[1]) is None:
                    file_dic[trunk[1]]=[trunk]
                else:
                    file_dic[trunk[1]].append(trunk)
                    
            file_word_distance_dic={}
            for (sent,file_name,sent_index,file_path) in distinct_ents: #get the summary of each file 
                if file_name not in file_summary_dic:
                    txt=''
                    with open(file_path, 'rb') as f:
                        txt=f.read()
                    txt=str(txt,encoding = "utf-8")
                    summary=summarize(txt,word_count=50)
                    
                    doc=nlp_2(txt)
                    cur_ent=nlp_2(ent_name)
#                     cur_ent=self.get_ent_from_doc(doc,ent_name)
                    if cur_ent is None:
                        print(ent_name)
                    file_word_distance={}
                    for token in doc:
                        if (not token.is_stop) and (token.pos_=='VERB') and (file_word_distance.get(token.text) is None) and token.lemma_!=ent_name:
                            file_word_distance[token.text]=cur_ent.similarity(token)
                    file_word_distance_list=[(word,distance) for (word,distance) in file_word_distance.items()]
                    file_word_distance_list.sort(key=lambda x:x[1], reverse=True)
                    file_summary_dic[file_name]=[summary,file_word_distance_list]       
                            
                        
                    
                    
                    
            file_name_list=list(file_dic.keys())  
            file_name_list.sort(reverse=False)
            
            output_txt='\n%d.  %s is found %d times in %d files.' % (ind+1,ent_name,len(ent_list),len(file_name_list))
            self.output_result.append(output_txt)
            print(output_txt)
#             print('%s is found in %d files' % (ent_name,len(file_name_list)))
            
            for (ind,file_name) in enumerate(file_name_list):
                trunk_list=file_dic.get(file_name)
                trunk_list.sort(key=lambda x:x[2])
                summary=file_summary_dic.get(file_name)[0]
                word_distance=['%s:%.3f' % (word,distance) for (word,distance) in file_summary_dic.get(file_name)[1][:10]]
                output_txt='(%d) File name: %s' % (ind+1,file_name)
                self.output_result.append(output_txt)
                print(output_txt)
                output_txt='    Summary: %s' % summary
                self.output_result.append(output_txt)
                print(output_txt)
                output_txt='    Words similarity: %s' % ','.join(word_distance)
                self.output_result.append(output_txt)
                print(output_txt)
                output_txt='    The trunk of the sentence containing the organization:'
                self.output_result.append(output_txt)
                print(output_txt)
                for (ind,trunk) in enumerate(trunk_list):
                    output_txt='    - %s (the %sth sentence)' % (trunk[0],trunk[2])
                    self.output_result.append(output_txt)
                    print(output_txt)
                    
                
        

# class chunk_entity(object):
#     '''
#     a chunk_entity object is a entity which is over one word
#     '''
#     def __init__(self,file_name=None,file_path=None,sent_index=None):
#         self.last_iob=None
#         self.ent=[]
#         self.file_name=file_name
#         self.sent_index=sent_index
#         self.file_path=file_path
# #         self.ent_type=None
#         self.verb=None
#         self.ent_complete=False
#         self.sent=None
#         self.root_dep=None
#         self.adj_list=[]
#         self.children=None
#         
#     def get_ent(self):
#         return ' '.join(self.ent)
#         
#     def complete_entity(self,token=None):
#         '''
#         complete entity chunk
#         '''
#         self.ent_complete=False
#         if token is not None:
#             ent_type=token.ent_type_
#             iob=token.ent_iob_
#             ent_text=token.lemma_
#             last_iob=None
# 
#             if ent_type == 'ORG' and iob in ['I','O','B']:
#                 if (iob=='B' and self.last_iob in [None,'O']) or (iob=='I' and self.last_iob in ['B','I']):           
#                     last_iob=iob
#                     self.ent.append(ent_text)
#                     self.sent=token.sent
#             #conduct whether the entity is completed
#             cond_1=self.last_iob in ['B','I']
#             cond_2=iob in ['O','B']
#             self.ent_complete=all([cond_1,cond_2])
#             if last_iob is not None: 
#                 self.last_iob=last_iob
#         return self.ent_complete
    
    

        

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
            yield file_name,file_path,str(article,encoding = "utf-8")

def get_parent_folder_path():
    return os.path.abspath(os.path.dirname(os.getcwd()))

def get_grand_parent_folder_path():
    return os.path.abspath(os.path.join(os.getcwd(), "../.."))

def get_abs_path(base_path,path_names):
    return os.path.join(base_path,path_names)


if __name__ == '__main__':
    folder_name='data'
    output_folder_name='result'
    output_file_name='statistic_result_test.txt'
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,folder_name)
    output_folder_path=get_abs_path(parent_folder_path,output_folder_name)
    output_file_path=get_abs_path(output_folder_path, output_file_name)
    nlp_1.remove_pipe("ner")  #remove ner module from pipe for speeding up
    ent_list=[]
    for (file_name,file_path,txt) in read_articles(folder_path):
        doc=nlp_1(txt)
        doc=nlp_2(doc._.coref_resolved)  #coreference resolution
        org_chunk=[chunk for chunk in doc.noun_chunks if chunk.root.ent_type_=='ORG']
        ent_list+=org_chunk
#         for (ind,sent) in enumerate(doc.sents):
#             cke=chunk_entity(file_name=file_name,file_path=file_path,sent_index=(ind+1))
#             for token in sent:              
#                 if cke.complete_entity(token=token):   
#                     ent_list.append(cke)
#                     cke=chunk_entity(file_name=file_name,file_path=file_path,sent_index=(ind+1))
#                     cke.complete_entity(token=token)
                    
    static_tool=static_tool()     
    for ent in ent_list: 
        static_tool.add_entity(ent)
    static_tool.output_static_results()
    static_tool.output_static_results_file(output_file_path)

    
    
    
    
    
    
    
    