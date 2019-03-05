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
            print(result)
            result.draw()
        return result

#     def processTree(self,tree):
#         prvWord=''
#         if tree and isinstance(tree, nltk.tree.Tree):
#             print(tree)
#             for node in tree:
#                 if isinstance(node, nltk.tree.Tree):
#                     self.processTree(node)
#                 else:
#                     if prvWord.upper()=='THE':
#                         val=self.countMap.get(node[0])
#                         if not val:
#                             val=1
#                         else:
#                             val=val+1
#                         self.countMap[node[0]]=val                        
#                     prvWord=node[0]
   
    def processTree(self,tree):
        
        if tree and isinstance(tree, nltk.tree.Tree):
            for node in tree:
                if isinstance(node, nltk.tree.Tree):
                    self.processTree(node)
                else:
                    if node[0].upper()=='THE':
                        self.buffer.append(node[0])
                    elif len(self.buffer)>0 and (node[1] in ('JJ','NN','NNP','CD','VBN')):
                        self.buffer.append(node[0])
                    else:
                        if len(self.buffer)>0:
                            str=' '.join(self.buffer)
                            val=self.countMap.get(str)
                            if not val:
                                val=1
                            else:
                                val=val+1
                            self.countMap[str]=val
                            self.buffer.clear()
        
    def getCountResult(self):
        aReturn=[item for item in self.countMap.items()]
        aReturn.sort(key=(lambda x:x[1]), reverse=False)
        return aReturn
    

if __name__ == '__main__':   
    sent = "The one thing missing in the debate about the proposed World League is any denial that the story broken by the Herald last week is true."
#     taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
    grammar = "NP: {<DT>?<JJ>*<NN>}"
#     cp = nltk.RegexpParser(grammar)
#     result = cp.parse(taggedS)
#     print(result)
#     result.draw()
    parser=TextParser(text=sent)
    result=parser.parseText(grammar)
    parser.processTree(result)
    print(parser.getCountResult())
    












