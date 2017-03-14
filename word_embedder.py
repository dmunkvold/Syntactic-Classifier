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
    
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        
        
    def build_graph():
        #sets up tensorflow graph
        
        self.graph = tf.Graph()
        with graph.as_default():
            
            # Input data.
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            
            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                embeddings = tf.Variable(
                        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) 
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)
                
                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                        tf.truncated_normal([vocabulary_size, embedding_size],
                                            stddev=1.0 / math.sqrt(embedding_size)))  
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
                
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.   
            loss = tf.reduce_mean(
                  tf.nn.nce_loss(weights=nce_weights,
                                 biases=nce_biases,
                                 labels=train_labels,
                                 inputs=embed,
                                 num_sampled=num_sampled,
                                 num_classes=vocabulary_size))
            
            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                  normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                  valid_embeddings, normalized_embeddings, transpose_b=True) 
            
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
                
        vocabulary_size = len(token_reps)
        counts = collections.Counter(token_strings).most_common(vocabulary_size - 1)
        reverse_token_reps = dict(zip(token_reps.values(), token_reps.keys()))      
        return tokens, counts, token_reps, reverse_token_reps
    
    
    def generate_batch(self, batch_size, num_skips, skip_window, tokens):
        #Function to generate a training batch for the skip-gram model.
        
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(tokens[self.token_index])
            self.token_index = (self.token_index + 1) % len(tokens)
        for i in range(batch_size // num_skips):
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
    
    
    def train_graph(self):
        #trains graph to produce token embeddings
        
        self.num_steps = 100001
        with tf.Session(graph=self.graph) as session:
            
            # We must initialize all variables before we use them.
            init.run()
            print("Initialized")
            
            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(
                        batch_size, num_skips, skip_window)  
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                
                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val
                
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                    
                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
                        
            final_embeddings = normalized_embeddings.eval()
            
    
    def generate_embedded_samples(self, text):
        tokenized_text = self.tokenize_text(text)
        #print tokenized_text
        cleaned_tokens, counts, token_reps, reverse_token_reps = self.clean_tokens(tokenized_text)
        self.token_index = 0
        for i in range(0, 10):
            batch, labels = self.generate_batch(batch_size=128, num_skips=2, skip_window=1, tokens=cleaned_tokens)
            print('BATCH', batch)
            print("LABLES", labels)
        #target_context = self.contextualize(cleaned_tokens)
        #print 'cleaned2', cleaned_tokens
        #token_chunks = self.chunker(cleaned_tokens)
        #samples, img_samples = self.build_samples(token_chunks, target_context)
        #return samples, img_samples