#!/usr/bin/env python
#coding: utf-8
'''
Created on Mar 5, 2019

@author: xingtong
'''
import nltk

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
        parse all text
        @param grammar:
        @return the result is parsed by nltk
        '''
        result=None
        if self.text and grammar:
            taggedS = nltk.pos_tag(nltk.word_tokenize(self.text))
            cp = nltk.RegexpParser(grammar)
            result = cp.parse(taggedS)
#             print(result)
            result.draw()
        return result
    
    def parseFile(self,grammar):
        '''
        pares file
        @param grammar:
        @return a iterator and the iterator can return each line parsed by nltk 
        '''
        result=None
        if self.filePath:
            f = open(self.filePath, "r")
            for line in f:
                if len(line.strip()):  #if the line is not empty
                    taggedS = nltk.pos_tag(nltk.word_tokenize(line))
                    cp = nltk.RegexpParser(grammar)
                    result = cp.parse(taggedS)
#                     result.draw()
                    yield result #yield each line
            f.close()
   
    def processTree(self,tree):
        '''
        The first method to process the tree structure
        '''
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


    def processTree2(self,tree):
        '''
        The second method to process the tree structure (I prefer this method:)
        '''
        if tree and isinstance(tree,nltk.tree.Tree):
            for subTree in tree.subtrees():
                if subTree and 'DNP'==subTree.label():
                    leaves=subTree.leaves()
                    if leaves[0][0] and leaves[0][0].upper()=='THE':
                        str=(' '.join([leaf[0] for leaf in leaves])).lower()
                        val=self.countMap.get(str)
                        if not val:
                            val=1
                        else:
                            val=val+1
                        self.countMap[str]=val
                        
        
    def getCountResult(self):
        '''
        get the result processed
        @return: a list containing the results
        '''
        aReturn=[item for item in self.countMap.items()]  #get the list containing all definite nouns
        aReturn.sort(key=(lambda x:x[1]), reverse=False)  #sort the list
        return aReturn
    
    def execute(self,grammar):
        '''
        execute
        '''
        for line in self.parseFile(grammar):  #parse each line
            self.processTree2(line)  #process the line (tree structure)
    

if __name__ == '__main__':
    sent=None
    filePath='test.txt'  # the text file path
    grammar = "DNP: {<DT>?(<RB>|<RBR>|<RBS>)*(<JJ>|<JJR>|<JJS>)*<IN>*(<NN>|<NNS>|<NNP>|<NNPS>)+}"
    parser=TextParser(text=sent,filePath=filePath)  #generate a object of TextParser
    parser.execute(grammar)
    results=parser.getCountResult()
    print('There are %d definite nouns:' % len(results))   #output the result
    for ind,item in enumerate(parser.getCountResult()):
        print('%d . %s : %s' % (ind+1,item[0],item[1]))













