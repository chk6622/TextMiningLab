import spacy
import neuralcoref
import subject_object_extraction
from textacy.extract import subject_verb_object_triples
from textacy.extract import semistructured_statements,named_entities,noun_chunks,acronyms_and_definitions
from gensim.summarization import summarize
from doc_topics import text_topics
# import textacy.extract
# from collections import Counter

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas import Series, DataFrame
from collections import Counter

import warnings
from audioop import reverse

def get_rights(word):
    s_return=[]
    for r in word.rights:
        s_return.append(r)
    return s_return

def get_mean_vector(sents):
    a=np.zeros(300)
    for sent in sents:
        a=a+sent.vector
    return a/len(sents)

def get_central_vector(sents):
    vecs=[]
    for sent in sents:
#         doc=nlp(sent)
        vecs.append(sent.vector)
    mean_vec=get_mean_vector(sents)
    index=pairwise_distances_argmin_min(np.array([mean_vec]),vecs)[0][0]
    return sents[index]


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
nlp = spacy.load("en_core_web_md")
doc = nlp(text)



vectors=[]
sentences=[]
indexes=[]
for (ind,sentence) in enumerate(doc.sents):
    indexes.append(ind)
    sentences.append(sentence)
    vectors.append(sentence.vector)
    
df=pd.DataFrame({'index':indexes,'sentence':sentences})
print(df.head(n=3))
    
x=np.array(vectors)
n_classes={}
for i in tqdm(np.arange(0.001,1,0.002)):
    dbscan=DBSCAN(eps=i,min_samples=2,metric='cosine').fit(x)
    n_classes.update({i:len(pd.Series(dbscan.labels_).value_counts())})
    
results=[(key,value) for key,value in n_classes.items()]
results.sort(key=lambda x:x[1],reverse=True)
print(results[0])

    
dbscan=DBSCAN(eps=results[0][0],min_samples=2,metric='cosine').fit(x)
count_result_tmp = Counter(dbscan.labels_)  
print(count_result_tmp)
count_result=[(cls,cont) for cls,cont in count_result_tmp.items() if cls!=-1]
results=pd.DataFrame({'label':dbscan.labels_,'sent':sentences})
print (results.describe())
nrow, ncol = results.shape
print ('%s rows, %s columns' % (nrow, ncol))
# print ("\n correlation Matrix")
for ind,count in count_result:
    sent_cluster = results[results.label==ind].sent.tolist()
    print(get_central_vector(sent_cluster))



























