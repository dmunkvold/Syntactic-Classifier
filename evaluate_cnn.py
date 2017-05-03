from word_embedder import WordEmbedder
from training_set_builder import SampleBuilder
import random

from bs4 import BeautifulSoup
import urllib


#urls is a list of string pairs [(url, category)....]
def convert_from_urls(urls):
    samples = []
    for url in urls:
        html = urllib.urlopen(url[0]).read()
        u = BeautifulSoup(html)
        text = u.get_text()
        word_embedder = WordEmbedder(16, 64, 2, 1)
        embeddings = word_embedder.generate_embeddings(text)
        height = 16
        for i in range(0, len(word_embedder.tokens)/height):
            pixel_data = []
            for j in range(0, height):
                pixel_data.append(embeddings[word_embedder.tokens[(i*height)+j]])
            #just saying 0 for now
            sample=(pixel_data, url[1])
            samples.append(sample)
        #print samples
    if len(samples) > 50 :
        samples = random.sample(samples, 50)
    shuffled_samples = scramble_samples(samples)
    data, labels = format_samples(shuffled_samples)
    return data, labels
    
def scramble_samples(samples):
    shuffled_samples = random.sample(samples, len(samples))
    return shuffled_samples
    
    
def format_samples(shuffled_samples):
    data = []
    labels = []
    for i in range(0, len(shuffled_samples)):
        data.append(shuffled_samples[i][0])
        labels.append(shuffled_samples[i][1])
    return data, labels


urls = [('https://en.wikipedia.org/wiki/Gentrification', 0)]    
data, labels = convert_from_urls(urls)
print data, labels