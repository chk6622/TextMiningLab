#!/usr/bin/env python
#coding: utf-8
'''
Created on Jun 3, 2019

@author: xingtong
'''
from audioop import reverse

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
import itertools
# import textacy.extract
# from collections import Counter
import warnings

warnings.filterwarnings('ignore')

nlp_1 = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"
neuralcoref.add_to_pipe(nlp_1)  #add NeuralCoref to the pipline of spacy to solve coreference resolution
nlp_2 = spacy.load('en_core_web_sm')
            
        
def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result

def extract_relations(doc,target_entity):
    rReturn=[]
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    relations = []
    for ent in doc:
        if ent.dep_ in ("attr", "dobj"):
            subject = [w for w in ent.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, subject.head.text, ent))
        elif ent.dep_ == "pobj" and ent.head.dep_ == "prep":
            relations.append((ent.head.head, ent.head.head.head.text,ent))
        elif ent.dep_=='pobj' and ent.head.dep_=='agent':
            relation=[]
            subject=None
            for w in ent.ancestors:
#                 print('%s-%s-%s' % (w,w.dep_,w.pos_))
                if w.dep_=='nsubj':
                    subject=w
                elif w.dep_!='ROOT':
                    relation.append(w.text)
            if subject:
                relation.sort(reverse=True)
                relations.append((subject,' '.join(relation), ent))
    for subj, relation, obj in relations:
        if target_entity is not None and target_entity in (subj.text,obj.text):
#             rReturn+="{:<10}({})\t{}\t{}\t{}({})\n".format(subj.text,subj.dep_, subj.head, obj.ent_type_, obj.text, obj.dep_)
            rReturn.append("   -{:<35}({:<5})({:<3})\t{:<15}\t{:<35}({:<5})({:<3})".format(subj.text, subj.dep_, subj.ent_type_, relation, obj.text, obj.dep_, obj.ent_type_))
    return rReturn
    

        

def read_articles(folder_path):
    '''
    read all articles in a folder
    @param folder: folder path
    @return: a list each item in the list is a article
    '''
    files= os.listdir(folder_path)
    size=len(files)
    article_list = []
    for ind,file_name in enumerate(files):
        article=None
        file_path=get_abs_path(folder_path, file_name)
        if not os.path.isdir(file_path):
            with open(file_path, 'rb') as f:
                article=f.read()
            yield ind,size,file_name,file_path,str(article,encoding = "utf-8")

def get_parent_folder_path():
    return os.path.abspath(os.path.dirname(os.getcwd()))

def get_grand_parent_folder_path():
    return os.path.abspath(os.path.join(os.getcwd(), "../.."))

def get_abs_path(base_path,path_names):
    return os.path.join(base_path,path_names)

def output_static_results_file(file_path,output_buffer):
        if not os.path.isdir(file_path):
            with open(file_path, 'w') as f:
                for line in output_buffer:
                    f.write(line+'\n')

def get_sentence_trunk(sentence):
    '''
    get all the subj+verb+obj phrase from the sentence 
    @param sentence:
    @return: a list, every item in the list is a subj+verb+obj phrase
    '''
    return [' '.join((str(sub),str(verb),str(obj))) for (sub,verb,obj) in subject_verb_object_triples(sentence)]

if __name__ == '__main__':
    output_buffer=[]
    folder_name='CCAT'
    output_folder_name='result'
    output_file_name='statistic_result_test.txt'
    parent_folder_path=get_parent_folder_path()
    folder_path=get_abs_path(parent_folder_path,folder_name)
    output_folder_path=get_abs_path(parent_folder_path,output_folder_name)
    output_file_path=get_abs_path(output_folder_path, output_file_name)
    nlp_1.remove_pipe("ner")  #remove ner module from pipe for speeding up
    ent_list=[]
    ent_dic={}
    print('Start counting the top-5 organizations.')
    for ind, size, file_name,file_path,text in read_articles(folder_path):   
#         if ind>50:
#             break    
        doc=nlp_1(text)
        doc=nlp_2(doc._.coref_resolved)  #coreference resolution
        org_chunk=[(chunk,file_name) for chunk in doc.noun_chunks if chunk.root.ent_type_=='ORG']
        ent_list+=org_chunk
        print('Processing {:4}/{:<4}.'.format(ind,size))
#     print(ent_list)
    for ent in ent_list:
        if ent_dic.get(ent[0].text) is None:
            ent_dic[ent[0].text]=[ent]
        else:
            val=ent_dic[ent[0].text]#=[ent_dic[ent.text]+1
            val.append(ent)
    ent_category_list=[(key,val) for (key,val) in ent_dic.items()]
#     print(ent_category_list)
    ent_category_list.sort(key=lambda x:len(x[1]),reverse=True)
#     tmp_list=[(ent,len(ent_list)) for (ent,ent_list) in ent_category_list[:5]]
#     print(tmp_list)
    ent_static_list=[]
    top_5=ent_category_list[:5]
    print('Finish counting the top-5 organizations.')
    print('Start searching relationship.')
    output_buffer.append('The top-5 organizations are %s.' % ', '.join([item[0] for item in top_5]))
    for ent_title,ent_list in top_5:
        doc_list=[]
        doc_dic={}
        sent_list=[]
        sent_dic={}
        
        for ent in ent_list:
            doc=ent[0].doc
            if doc_dic.get(doc.text) is None:
                doc_list.append((doc,ent[1]))
                doc_dic[doc.text]=''
            
            sent=ent[0].sent
            if sent_dic.get(sent.text) is None:
                sent_list.append(sent)
                sent_dic[sent.text]=''
                
            
        ent_static_list.append([ent_title,doc_list,sent_list,ent_list])
    for ind,(ent_title,doc_list,sent_list,ent_list) in enumerate(ent_static_list):
        output_buffer.append('\n%d. %s is found %d times in %d files.' % (ind+1,ent_title,len(ent_list),len(doc_list)))
        
        for ind,(doc,doc_name) in enumerate(doc_list):
            output_buffer.append(' (%d). %s' % (ind+1,doc_name))
            for sent in sent_list:
                if sent.doc==doc:
#                     trunk=get_sentence_trunk(sent)
                    trunk=extract_relations(nlp_2(sent.text),ent_title)
                    if trunk is not None and len(trunk)>0:
                        output_buffer+=trunk
#                     else:
#                         output_buffer.append(sent.text)
    output_static_results_file(output_file_path,output_buffer)
    print('Finish searching relationship.')
#                         for line in trunk:
#                             print(line)
#                     else:
#                         print(sent)
#                 print(sent)
#                 if sent.doc==doc:
#                     sent_ent=[ent for ent in sent.ents if ent.text==ent_title]
#                     right_words=[]
#                     left_words=[]
#                     key_words=[]
#                     for ent in sent_ent:
#                         right_words=list(itertools.islice(ent.rights, 3))
#                         left_words=list(itertools.islice(ent.lefts, 3))
#                     key_words+=[item.text for item in left_words]
#                     key_words.append(ent_title)
#                     key_words+=[item.text for item in right_words]
# #                     print(key_words)
#                     print('----%s' % ' '.join(key_words))
#             file_word_distance={}
#             
#             cur_ent=ent_list[0]
#             for token in doc:
#                 if (not token.is_stop) and (token.pos_=='VERB') and (file_word_distance.get(token.text) is None) and token.lemma_!=ent_title:
#                     try:
#                         file_word_distance[token.text]=cur_ent.similarity(token)
#                     except Exception:
#                         pass
            
#             dep_dic={}
#             for ent in ent_list:   
#                 if ent.doc==doc:
#                     dep=ent.root.dep_
#                     dp=dep_dic.get(dep)
#                     if dp is None:
#                         dep_dic[dep]=1
#                     else:
#                         dep_dic[dep]+=1
#             file_word_distance_list=[(word,distance) for (word,distance) in file_word_distance.items()]
#             file_word_distance_list.sort(key=lambda x:x[1], reverse=True)
#             print(file_word_distance_list[:5])
#             print(dep_dic)
#         print()
    
    
    
    
    
    
    
    