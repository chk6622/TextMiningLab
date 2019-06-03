import tensorflow
import pandas as pd
#Deinfine the tensor flow graph. That is define the NN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

corpus = ['king is a strong man',
          'queen is a wise woman',
          'boy is a young man',
          'girl is a young woman',
          'prince is a young king',
          'princess is a young queen',
          'man is strong',
          'woman is pretty',
          'prince is a boy will be king',
          'princess is a girl will be queen']


# Define a method to remove stop words
def remove_stop_words(corpus):
    stop_words = ['is', 'a', 'will', 'be']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))

    return results


corpus = remove_stop_words(corpus)
words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)

# Generate context for each word in a defined window size
word2int = {}

for i, word in enumerate(words):
    word2int[word] = i

# print(word2int)

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
# print(sentences)

WINDOW_SIZE = 1

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0): min(idx + WINDOW_SIZE, len(sentence)) + 1]:
            if neighbor != word:
                data.append([word, neighbor])

# Now create an input and a output label as reqired for a machine learning algorithm.
for text in corpus:
    print(text)





df = pd.DataFrame(data, columns=['input', 'label'])
# print(df)
# print(df.shape)
# print(word2int)




ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
# He is a teacher -> he :0, teacher: 1; ONE_HOT_DIM=2
# to_one_hot_encoding(0)-> [1,0]
# to_one_hot_encoding(1)-> [0,1]
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # input word
Y = [] # target word

# print(df['input'])

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))

# convert them to numpy arrays
# print(X)
X_train = np.asarray(X) #array[][]->matrix
# print(X_train)
Y_train = np.asarray(Y) #array[][]->matrix

# making placeholders for X_train and Y_train, tf:tensorflow
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM)) 
# print(x)
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
# print(y_label)

# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 2


# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#Train the NN
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 5000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))


# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
#print(vectors)

#Print the word vector in a table
w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
w2v_df['word'] = words
w2v_df = w2v_df[['word', 'x1', 'x2']]
# print(w2v_df)

#Now print the word vector as a 2d chart


fig, ax = plt.subplots()
 
for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
    ax.annotate(word, (x1, x2))
 
PADDING = 1.0
x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
 
plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (10, 10)
 
plt.show()






'''
question 4.
Add in code to compute the distance of each of the words to each of the other words. 
You can use the dot product to calculate the cosine distance between each of the words.
'''

def get_cosine_distance(vector1,vector2):
    '''
    get cosine distance between vector1 and vector2
    '''
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

cosine_distance_list=[]
for row_1 in vectors:
    vector1 = np.array(row_1)
    row_cosine_distance=[]
    for row_2 in vectors:
        vector2 = np.array(row_2)
        cosine_distance=get_cosine_distance(vector1,vector2)
        row_cosine_distance.append(cosine_distance)
    cosine_distance_list.append(row_cosine_distance)    
# cosine_distance_matrix = np.asarray(cosine_distance_list)
# print(words)
word_list=list(words)
cosine_distance_df = pd.DataFrame(cosine_distance_list, columns = word_list)
cosine_distance_df['word'] = words
output_columns=[]
output_columns.append('word')
output_columns+=word_list
# cosine_distance_df = cosine_distance_df[output_columns]
print('\nThe distance of each of the words to each of the other words:')
print(cosine_distance_df[output_columns])
'''
The distance of each of the words to each of the other words:
        word      wise     queen  ...    pretty     woman      girl
0       wise  1.000000 -0.056706  ...  0.837126 -0.129328  0.263118
1      queen -0.056706  1.000000  ... -0.593599  0.997340  0.948291
2      young  0.620260 -0.818307  ...  0.948308 -0.858026 -0.593556
3     prince -0.762667  0.688998  ... -0.992250  0.739993  0.423329
4        boy -0.937647 -0.293859  ... -0.594795 -0.223406 -0.582052
5   princess -0.239499  0.982915  ... -0.731581  0.993717  0.873670
6        man -0.985155  0.227257  ... -0.918604  0.297635 -0.093592
7       king -0.955044  0.350143  ... -0.961661  0.417488  0.034728
8     strong -0.093691 -0.988686  ...  0.466172 -0.975123 -0.985172
9     pretty  0.837126 -0.593599  ...  1.000000 -0.650680 -0.307472
10     woman -0.129328  0.997340  ... -0.650680  1.000000  0.922633
11      girl  0.263118  0.948291  ... -0.307472  0.922633  1.000000
'''


'''
question 5.
Output 2 nearest neighbours
'''
nearest_neighbours={}
for cosine_distance_row in cosine_distance_list:
    word_to_word_cosine_distance=[(word,val) for (word,val) in zip(word_list,cosine_distance_row)]
    word_to_word_cosine_distance.sort(key=lambda x:abs(x[1]), reverse=True)
    nearest_neighbours[word_to_word_cosine_distance[0][0]]=[word_to_word_cosine_distance[1][0],word_to_word_cosine_distance[2][0]]
print('\n2 nearest neighbours:')
for key in nearest_neighbours.keys():
    print('%s: %s, %s' % (key,nearest_neighbours.get(key)[0],nearest_neighbours.get(key)[1]))
'''
2 nearest neighbours:
wise: man, king
queen: woman, strong
young: prince, pretty
prince: pretty, young
boy: wise, man
princess: woman, queen
man: king, wise
king: man, pretty
strong: queen, girl
pretty: prince, king
woman: queen, princess
girl: strong, queen
'''   
    
'''
question 6.
Change the number of embeddings in the dimensions to 4 and now calculate the distance for each of the words to the other words.
'''
# word embedding will be 2 dimension for 2d visualization
EMBEDDING_DIM = 4


# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#Train the NN
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iteration = 5000
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 3000 == 0:
        print('iteration '+str(i)+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))


# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
#print(vectors)

#Print the word vector in a table
w2v_df = pd.DataFrame(vectors, columns = ['x1','x2','x3','x4'])
w2v_df['word'] = words
# print(w2v_df[['word', 'x1','x2','x3','x4']])

cosine_distance_list=[]
for row_1 in vectors:
    vector1 = np.array(row_1)
    row_cosine_distance=[]
    for row_2 in vectors:
        vector2 = np.array(row_2)
        cosine_distance=get_cosine_distance(vector1,vector2)
        row_cosine_distance.append(cosine_distance)
    cosine_distance_list.append(row_cosine_distance)    
# cosine_distance_matrix = np.asarray(cosine_distance_list)
# print(words)
word_list=list(words)
cosine_distance_df = pd.DataFrame(cosine_distance_list, columns = word_list)
cosine_distance_df['word'] = words
output_columns=[]
output_columns.append('word')
output_columns+=word_list
# cosine_distance_df = cosine_distance_df[output_columns]
print('\nChange the number of embeddings in the dimensions to 4.')
print('\nThe distance of each of the words to each of the other words:')
print(cosine_distance_df[output_columns])
'''
The distance of each of the words to each of the other words:
        word      wise     queen  ...    pretty     woman      girl
0       wise  1.000000 -0.689126  ...  0.679620 -0.784636  0.029812
1      queen -0.689126  1.000000  ...  0.010618  0.402790 -0.482971
2      young  0.199427 -0.338146  ...  0.250891 -0.542436  0.301040
3     prince -0.219368  0.287946  ...  0.171978 -0.406938  0.529451
4        boy -0.511156 -0.243541  ... -0.813614  0.459205  0.505632
5   princess -0.786660  0.754714  ... -0.168983  0.317633 -0.162262
6        man -0.209990  0.334051  ... -0.086130  0.007359  0.413614
7       king -0.055872  0.363671  ...  0.278483 -0.365354  0.389401
8     strong  0.156968 -0.646966  ... -0.335395  0.150451  0.033558
9     pretty  0.679620  0.010618  ...  1.000000 -0.814772 -0.345713
10     woman -0.784636  0.402790  ... -0.814772  1.000000 -0.234662
11      girl  0.029812 -0.482971  ... -0.345713 -0.234662  1.000000

'''


'''
question 7.
7.Again find the 2 nearest context words for each of the words.
'''
nearest_neighbours={}
for cosine_distance_row in cosine_distance_list:
    word_to_word_cosine_distance=[(word,val) for (word,val) in zip(word_list,cosine_distance_row)]
    word_to_word_cosine_distance.sort(key=lambda x:abs(x[1]), reverse=True)
    nearest_neighbours[word_to_word_cosine_distance[0][0]]=[word_to_word_cosine_distance[1][0],word_to_word_cosine_distance[2][0]]
print('\n2 nearest neighbours:')
for key in nearest_neighbours.keys():
    print('%s: %s, %s' % (key,nearest_neighbours.get(key)[0],nearest_neighbours.get(key)[1]))
'''
2 nearest neighbours:
wise: princess, woman
queen: princess, wise
young: woman, man
prince: king, strong
boy: pretty, strong
princess: wise, queen
man: king, strong
king: strong, man
strong: king, man
pretty: woman, boy
woman: pretty, wise
girl: prince, boy
'''