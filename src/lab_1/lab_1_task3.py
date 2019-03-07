import nltk
sent = "I in the park saw a boy with a telescope."
taggedS = nltk.pos_tag(nltk.word_tokenize(sent))
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(taggedS)
print(result)
result.draw()

# I changed the sentence to 
# ‘I in the park saw a boy with a telescope.’, 
# and use the same grammar to parse the sentence. the result is:
# (S
#   I/PRP
#   in/IN
#   (NP the/DT park/NN)
#   saw/VBD
#   (NP a/DT boy/NN)
#   with/IN
#   (NP a/DT telescope/NN)
#   ./.)
# I find that if we have the same grammar 
# we can get the different tree with the same 
# meaning no matter how to rewrite the 
# sentence in a different way. 