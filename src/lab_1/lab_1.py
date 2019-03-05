#!/usr/bin/env python
#coding: utf-8
'''
Created on Mar 5, 2019

@author: xingtong
'''
import nltk
import string

class TextParser(object):
    '''
    the class is to process text document using NLTK
    '''
    def __init__(self,text=None,filePath=None):
        self.text=text
        self.filePath=filePath
        self.buffer=[]
        self.countMap={}
        
    def parseText(self,grammar):
        '''
        
        '''
        result=None
        if self.text and grammar:
            taggedS = nltk.pos_tag(nltk.word_tokenize(self.text))
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(taggedS)
#             print(result)
#             result.draw()
        return result
    
    def parseFile(self,grammar):      
        result=None
        if self.filePath:
            f = open(self.filePath, "r")
            for line in f:
                if len(line.strip()):  #if the line is not empty
                    taggedS = nltk.pos_tag(nltk.word_tokenize(line))
                    cp = nltk.RegexpParser(grammar)
                    result = cp.parse(taggedS)
                    yield result
        f.close()
   
    def processTree(self,tree):
        
        if tree and isinstance(tree, nltk.tree.Tree):
#             print(tree.leaves())
            for node in tree.leaves():
                if node[0].upper()=='THE':
                    if len(self.buffer)>0:
                            str=(' '.join([item[0] for item in self.buffer])).lower()
                            val=self.countMap.get(str)
                            if not val:
                                val=1
                            else:
                                val=val+1
                            self.countMap[str]=val
                            self.buffer.clear() 
                    self.buffer.append(node)
                elif len(self.buffer)>0 and ('NN' in self.buffer[-1:][0][1]) and ('NN' not in node[1]):#(node[1] in ('JJ','NN','NNP','CD','VBN')):              
                        if len(self.buffer)>0:
                            str=(' '.join([item[0] for item in self.buffer])).lower()
                            val=self.countMap.get(str)
                            if not val:
                                val=1
                            else:
                                val=val+1
                            self.countMap[str]=val
                            self.buffer.clear()
                else:
                    if len(self.buffer)>0:
                        self.buffer.append(node)
#             print(self.buffer)
        
    def getCountResult(self):
        aReturn=[item for item in self.countMap.items()]
        aReturn.sort(key=(lambda x:x[1]), reverse=False)
        return aReturn
    

if __name__ == '__main__':   
    sent = '''Vegetarianism is on the rise, as many vegetarians will gladly tell you.

While many people who eschew meat products do so for the sake of animals and the environment, we're starting to learn more about the negative health effects of meat and the benefits from eating a plant-based diet.

We asked five experts if a vegetarian diet is healthier.'''
#     taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
    filePath='test.txt'
    grammar = "NP: {<DT>?<JJ>*<NN>}"
#     cp = nltk.RegexpParser(grammar)
#     result = cp.parse(taggedS)
#     print(result)
#     result.draw()
    parser=TextParser(text=sent,filePath=filePath)
#     result=parser.parseText(grammar)
#     print(result.leaves())
    for line in parser.parseFile(grammar):
        parser.processTree(line)
    for ind,item in enumerate(parser.getCountResult()):
        print('%d . %s : %s' % (ind+1,item[0],item[1]))
#     str='a'
#     print('d' in str)
#     print(str[-1:])












