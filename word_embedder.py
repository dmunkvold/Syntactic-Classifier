import pattern
from pattern.en import parse
import re
import math
import collections
import numpy as np
import random
import tensorflow as tf
#from PIL import Image

class WordEmbedder():
    
    def __init__(self, batch_size, embedding_size, skip_window, num_skips):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips        
        
        
    def build_graph(self):
        #sets up tensorflow graph
        
        #the batch size variable needs to be elsewhereFIXTHIS
        self.valid_size = 16     # Random set of words to evaluate similarity on.
        valid_window = 32  # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.random.choice(valid_window, self.valid_size, replace=False)
        num_sampled = 32         
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            
            # Input data.
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
            
            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)) 
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)
                
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                        tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))  
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
                
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.   
            self.loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=nce_weights,
                                 biases=nce_biases,
                                 labels=self.train_labels,
                                 inputs=embed,
                                 num_sampled=num_sampled,
                                 num_classes=self.vocabulary_size))
            
            # Construct the SGD optimizer using a learning rate of 1.0.
            self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
            
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                  self.normalized_embeddings, valid_dataset)
            self.similarity = tf.matmul(
                  valid_embeddings, self.normalized_embeddings, transpose_b=True) 
            
            # Add variable initializer.
            init = tf.global_variables_initializer()
            
            
    def tokenize_text(self, text):
        #extract syntactic components of words
        
        tokenized_text = parse(text)
        return tokenized_text
    
    
    def clean_tokens(self, tokenized_text):
        #eliminates semantic component of word-tokens and assigns a unique integer
        #to represent each syntactic token.
        
        token_reps = {}
        tokens = []
        token_strings = []
        words = re.split(" |\n", tokenized_text)
        count = 0
        for word in words:
            
            #trim symbols to remove the actual words, only leaving syntactic components
            #There's porbably a better way to do this bit down here in the next 4 lines:
            for i in range(0, len(word)-1):
                if word[i] == '/':
                    word = word[(i+1):]
                    token_strings.append(word)
                    break   
                
            #assign integer value to each token
            if word not in token_reps.keys():
                token_reps[word] = count
                tokens.append(count)
                count += 1
            else:
                tokens.append(token_reps[word])
                
        self.vocabulary_size = len(token_reps)
        self.counts = collections.Counter(token_strings).most_common(self.vocabulary_size - 1)
        self.reverse_token_reps = dict(zip(token_reps.values(), token_reps.keys())) 
        self.token_reps = token_reps
        self.tokens = tokens
        print ('LENGTH', len(tokens))
        #return tokens, self.counts, token_reps, reverse_token_reps
    
    
    def generate_batch(self, num_skips, skip_window, tokens):
        #Function to generate a training batch for the skip-gram model.
        
        assert self.batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(tokens[self.token_index])
            self.token_index = (self.token_index + 1) % len(tokens)
        for i in range(self.batch_size // num_skips):
            target = skip_window  # target label at the center of the buffer
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(tokens[self.token_index])
            self.token_index = (self.token_index + 1) % len(tokens)
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.token_index = (self.token_index + len(tokens) - span) % len(tokens)
        return batch, labels    
    
    
    def train_graph(self, num_steps):
        #trains graph to produce token embeddings
                  
        self.num_steps = num_steps
        
        with tf.Session(graph=self.graph) as session:
            
            # We must initialize all variables before we use them.
            init=tf.global_variables_initializer()
            init.run()
            print("Initialized")
            
            average_loss = 0
            for step in xrange(self.num_steps):
                batch_inputs, batch_labels = self.generate_batch(
                        self.num_skips, self.skip_window, self.tokens)  
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
                
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += loss_val
        
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                   
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    
                    sim = self.similarity.eval()

                    for i in xrange(self.valid_size):
                        
                        valid_word = self.reverse_token_reps[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverse_token_reps[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
                        
            final_embeddings = self.normalized_embeddings.eval()
            print final_embeddings
            return final_embeddings
            
    
    def generate_embeddings(self, text):
        tokenized_text = self.tokenize_text(text)
        #print tokenized_text
        self.clean_tokens(tokenized_text)
        self.token_index = 0
        self.build_graph()
        final_embeddings = self.train_graph(10001)
        print final_embeddings[0]
        return final_embeddings
        #for i in range(0, 10):
        #    batch, labels = self.generate_batch(batch_size=128, num_skips=2, skip_window=1, tokens=cleaned_tokens)
        #    print('BATCH', batch)
        #    print("LABLES", labels)
        #target_context = self.contextualize(cleaned_tokens)
        #print 'cleaned2', cleaned_tokens
        #token_chunks = self.chunker(cleaned_tokens)
        #samples, img_samples = self.build_samples(token_chunks, target_context)
        #return samples, img_samples