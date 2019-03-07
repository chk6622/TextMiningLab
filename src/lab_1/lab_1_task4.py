import nltk
sent = "There is a beautiful ball under the tree."
taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(taggedS)
print(result)
result.draw()

#I used the program to parse the sentence 
# ‘There is a beautiful ball under the tree.’ 
# and get the tree as following:
# (S
#   There/EX
#   is/VBZ
#   (NP a/DT beautiful/JJ ball/NN)
#   under/IN
#   (NP the/DT tree/NN)
#   ./.)
# I think the parser is correct.