#!/usr/bin/env python
#coding: utf-8
'''
Created on Jun 6, 2019

@author: xingtong
'''
from audioop import reverse
'''
1.得到最相近的名词top-5
2.含有这些名词的句子
'''
import spacy
import neuralcoref
import subject_object_extraction
from textacy.extract import subject_verb_object_triples
from textacy.extract import semistructured_statements
from gensim.summarization import summarize
from doc_topics import text_topics
# import textacy.extract
# from collections import Counter
import warnings

warnings.filterwarnings('ignore')

def get_sentence_trunk(sentence):
    '''
    get all the subj+verb+obj phrase from the sentence 
    @param sentence:
    @return: a list, every item in the list is a subj+verb+obj phrase
    '''
    return [' '.join((str(sub),str(verb),str(obj))) for (sub,verb,obj) in subject_verb_object_triples(sentence)]

def get_subject(subj):
    subject=[]
    for left in subj.lefts:
        subject.append(left.lemma_)
    subject.append(subj.lemma_)
    for right in subj.rights:
        subject.append(right.lemma_)
    return ' '.join(subject)

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
#         print(subj.text,obj.text)
        if target_entity is not None and ((subj.lemma_ in target_entity) or (obj.lemma_ in target_entity)):
            subject=get_subject(subj)
            object=get_subject(obj)
            rReturn.append("   -{:<55}\t{:<15}\t{:<55}".format(subject, relation, object))
    return rReturn



nlp_1 = spacy.load('en_core_web_sm')           # load model package "en_core_web_sm/md"
neuralcoref.add_to_pipe(nlp_1)  #add NeuralCoref to the pipline of spacy to solve coreference resolution
nlp_2 = spacy.load('en_core_web_sm')


text='''
Chrysler Corp. Tuesday announced $380 million in new investments for South America, including assembly plants for pickup trucks and diesel engines in Brazil and the expansion of a Jeep plant now being built in Argentina.
Chrysler, which is cautiously trying to rebuild its international presence, said the projects in Brazil were worth about $315 million, and the expansion in Argentina was worth about $65 million.
Roughly one third of the total investment, or about $126.6 million, will come from Chrysler's suppliers, who will play a major role in the automaker's low-risk global growth strategy, Chrysler Chairman Robert Eaton said.
"We don't intend to make risky investments just to be a major player in emerging markets," added Thomas Gale, Chrysler's executive vice president of international operations. "We're quite content to grow at a steady pace in regions where we see solid opportunities."
Eaton said the investments will boost Chrysler's sales in the Mercosur free-trade zone, which groups Argentina, Brazil, Paraguay and Uruguay.
But the company's limited production capacity will allow it only to grab a small portion of the Mercosur market away from rivals General Motors Corp., Ford Motor Co. and Volkswagen AG, he said.
"We are targeting very specific market segments," he said. "We don't have any interest or desire to offer a vehicle for every possible application."
The new Brazilian plant, which will be Chrysler's third limited-production facility in South America, will assemble the automaker's all-new Dakota compact pickup truck for sale in Argentina, Brazil, Paraguay and Uruguay, the countries in the Mercosur free-trade zone.
In Argentina, Chrysler said it will add production of about 6,000 Jeep Cherokees a year at a plant now under construction in the Cordoba province. The plant is already scheduled to build about 14,000 Jeep Grand Cherokees per year starting next April, and Cherokee output will begin in 1998.
A site for the Brazilian plant will be selected by year-end and vehicles will roll off the assembly line starting in mid-1998, Chrysler said. Production, however, will be modest, with 12,000 trucks in the first year and an ultimate capacity of 40,000 units annually. Employment will start at 400 people.
The trucks at first will be largely assembled from "complete knock-down" kits shipped from the United States, but the automaker intends to meet the Mercosur trade bloc's 60 percent local content requirement after three years.
Chrysler has not decided whether to market the Dakota under the Dodge brand name or under one of its other brands. The automaker now uses the only Jeep and Chrysler brand names outside the United States, Canada and Mexico.
The $315 million Brazil investment also includes a new diesel engine plant to be built by Detroit Diesel Corp..
The $10 million facility will supply the company's Italian-designed VM Motori four-cylinder turbocharged diesel engines for use in the Brazilian Dakota as well as in Jeep models built in Argentina. Chrysler installs about 40,000 of the engines annually into minivans and Jeep Grand Cherokees sold in Europe.
Others suppliers supporting the Chrysler by opening plants in Latin America include Dana Corp., Johnson Controls Inc., Lear Corp., Lear Corp. United Technologies Corp. and PPG Industries Inc., Chrysler executives said.
Eaton said total annual vehicle sales in the four-country Mercosur region will increase from about 2 million units currently to about 2.5 million by the end of the decade.
"We think this is a major growth area," Eaton said. "It's politically and economically a stable region, we think with particularly rising consumer buying power."
Including a small plant in Venezuela that assembles Cherokees and Neon small cars from kits, the investments announced Tuesday bring to $735 million the total financial commitments Chrysler and its suppliers have made in South America, the company said.
Chrysler stock rose 25 cents to close at $28.875 Tuesday on the New York Stock Exchange.'''

doc=nlp_1(text)

doc=nlp_2(doc._.coref_resolved)  #coreference resolution

# for sent in doc.sents:
#     print(sent)
#     print([(token.text,token.pos_.lower()) for token in sent if ('adj' not in token.pos_.lower()) or ('adv' not in token.pos_.lower())])
#     print('adj' not in 'bbadjaaa')

file_word_distance={}
ent_list=[]          
org_chunk=[(chunk,'file_name') for chunk in doc.noun_chunks]# if chunk.root.ent_type_=='ORG']
ent_list+=org_chunk
#将简写实体名转换为全称
new_ent_list=[]
for ent_out in ent_list:
#     outer=ent_out[0].lemma_
#     temp=None
#     if ent_out[0].root.ent_type_ in ('ORG','PERSON'):
#         for ent_in in ent_list:
#             inner=ent_in[0].lemma_
#             if inner !=outer and (outer in inner):
#                 temp=ent_in
#                 print('%s---->%s' % (outer,inner))
#     if temp is not None:
#         new_ent_list.append((temp[0],ent_out[1],ent_out[0].sent)) #(ent,file_name,sentence)
#     else:
        new_ent_list.append((ent_out[0],ent_out[1],ent_out[0].sent))
# print(new_ent_list)
#合并ent
new_ent_dic={}
for item in new_ent_list:
    ent=item[0]
    file_name=item[1]
    key=ent.text+file_name
    value=new_ent_dic.get(key)
    if value is None:
        new_ent_dic[key]=[item[0],item[1],[item[2],],1]
    else:
        if item[2] not in value[2]:
            value[2].append(item[2])
        value[3]=value[3]+1

new_ent_list_distinct=[(value[0],value[1],value[2],value[3]) for key,value in new_ent_dic.items()]
new_ent_list_distinct.sort(key=lambda x:x[3],reverse=True) #(ent,file_name,sentence,count)
# print(new_ent_list_distinct) 
#统计出现最多的organization
org_count={}
for item in new_ent_list_distinct:
    entity=item[0]
#     file_name=item[1]
#     sentence=item[2]
    count=item[3]
    if entity.root.ent_type_!='ORG':
        continue
    value=org_count.get(entity)
    if value is None:
        org_count[entity]=count
    else:
        org_count[entity]=value+count
org_list=[(key,value) for (key,value) in org_count.items()]
org_list.sort(key=lambda x:x[1],reverse=True)
# print(org_list)
top_5_ent=[org for org,count in org_list[:5]]
# print(top_5_ent)

#统计与organization最近似的组织,包括org本身
cur_ent=top_5_ent[0]
for chunk in new_ent_list_distinct:
    if (file_word_distance.get(chunk[0].text) is None):# and chunk[0].text!=cur_ent.text:
        try:
            similarity=cur_ent.similarity(chunk[0])
#             print(temp)
            file_word_distance[chunk[0]]=(chunk[1],chunk[2],similarity)
        except Exception as ex:
            print(ex)
file_word_distance_list=[((key,value[0],value[1],value[2])) for (key,value) in file_word_distance.items()] #(ent,file_name,sentence,similarity)
# print(file_word_distance_list)
file_word_distance_list.sort(key=lambda x:x[-1],reverse=True)
top_5_similarity=[a for (a,b,c,d) in file_word_distance_list[:10]]
# print(top_5_similarity)
# print('--------------------------------------------------------------')
# relations=extract_relations(doc, top_5_similarity)
# for relation in relations:
#     print(relation)
# for sent in doc.sents:
    





# sentence_set=set()
# for entity in top_5_similarity:
#     sentences=entity[-1]
#     for sentence in sentences:
#         sentence_set.add(sentence)
# for sentence in sentence_set:
#     print(sentence)

# target_ent=[]
# target_ent+=top_5_Similarity
# target_ent.append((cur_ent,)
# print(target_ent)
def fun(sent,target_ent):
    b_return=False
    for ent in target_ent:
        if ent.lemma_ in sent.lemma_:
            b_return=True
            break
    return b_return

def fun2(word,target_words):
    b_return=False
    for target_word in target_words:
        if word in target_word.lemma_:
            b_return=True
            break
    return b_return


#
trunks=[]
for (ind,sent) in enumerate(doc.sents):
#     print(sent)
    if fun(sent,top_5_similarity):
#         print(sent)
        sentence=[]
        ents=[ent.lemma_ for ent in sent.ents]
        trks=[ent.lemma_ for ent in sent.noun_chunks]
#         print(ents)
#         print(trks)
        for token in sent:
#             if ('obj' in token.pos_) or ('adv' in token.pos_):
#                 continue
            if fun2(token.lemma_,top_5_similarity) or token.pos_ in ('VERB','CCONJ','CONJ','NOUN') or token.lemma_.lower() in ('not','no'):#,'CCONJ','NOUN'):
                sentence.append(token.text)
            
        trunks.append(sentence)
        print('%d.%s' % (ind+1,sent))
        print('---%s' % ' '.join(sentence))
# for trunk in trunks:
#     print(' '.join(trunk))
#         trunks+=get_sentence_trunk(sent)
# for trunk in trunks:
#     print(trunk)
    
# dep_dic={}
# for ent in ent_list:   
#     if ent.doc==doc:
#         dep=ent.root.dep_
#     dp=dep_dic.get(dep)
#     if dp is None:
#         dep_dic[dep]=1
#     else:
#         dep_dic[dep]+=1
# file_word_distance_list=[(word,distance) for (word,distance) in file_word_distance.items()]
# file_word_distance_list.sort(key=lambda x:x[1], reverse=True)
# print(file_word_distance_list[:5])
# print(dep_dic)

















