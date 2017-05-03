from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder
from image_generator import ImageGenerator
from PIL import Image
import random
import numpy
import glob

class SampleBuilder():
    
    def __init__(self, categories):
        self.categories = categories
    
    def create_folders():
        for i in categories:
            newpath = r'./lib/'+ i 
            if not os.path.exists(newpath):
                os.makedirs(newpath)        
        
    def generate_samples_from_pdf(self, pdf, category):
        pdf_extractor = PDFExtractor()
        folder = r'./lib/samples/' + category + '-eval'
        word_embedder = WordEmbedder(32, 64, 3, 1)            
        text = pdf_extractor.extract_text(pdf)
        tag = pdf[-8:-4]
        self.embeddings = word_embedder.generate_embeddings(text)
            #img_gen = ImageGenerator(word_embedder.tokens, embeddings)
        self.generate_samples(32, folder, tag, word_embedder.tokens)
        
    #The two functions below are no longer up to date
    def convert_pdfs(self):
        for i in self.categories:
            folder = './lib/pdfs/' + i + '-eval'
            for filename in glob.glob(folder + '/*.pdf'): 
                print filename
                self.generate_samples_from_pdf(filename, i)
        #for j in pdfs.keys():
        #    assert j in self.categories
        #    self.generate_samples_from_pdfs(pdfs[j], j)
    
    
    def fetch_from_folder(self, category):
        image_list = []
        for filename in glob.glob('./lib/' + category + '4/*.png'): 
            im=Image.open(filename)
            image_list.append(im)        
    
    #def generate_set(self):
    #    samples = [] # first as (sample, label) tuples
    #    for j in self.categories:
    
    
    def generate_samples(self, sample_length, folder, tag, tokens):
        height = sample_length
        for i in range(0, len(tokens)/height):
            pixel_data = []
            for j in range(0, height):
                pixel_data.append(self.embeddings[tokens[(i*height)+j]])
            #norm = self.normalize(pixel_data)
            numpy.savetxt(folder + '/' + tag+str(i), pixel_data)
            
            #potential to speed up process of pixel data to training set by avoiding saving it as image. Right
            # now I am going to save as txt and then grab the data from the txt 
            
    
    def normalize(self, pixel_data):
        pd = numpy.array(pixel_data)
        norm_pd = numpy.zeros((64, 64), numpy.uint8)
        max_value = numpy.amax(pd)
        min_value = numpy.amin(pd)
        val_range = max_value + -min_value
        for i in range(0, len(pd)-1):
            for j in range(0, len(pd[i])-1):
                val = ((pd[i][j] + -min_value)/val_range)*255
                norm_pd[i][j] = val
                    
        print "after", norm_pd
        return norm_pd  
    #editing this for 16 instead of 32 size 
    def load_dataset_from_disk(self):
        samples = []
        for i in range(0, len(self.categories)):
            folder = './lib/samples/' + self.categories[i] + '-eval/'
            for filename in glob.glob(folder + '*'): 
                sample = numpy.loadtxt(filename, dtype=numpy.float32)
                sam = numpy.split(sample, 2)
                samples.append((sam[0], i))
                samples.append((sam[1], i))
        shuffled_samples = self.scramble_samples(samples)
        data, labels = self.format_samples(shuffled_samples)
        return data, labels
        
                
    def scramble_samples(self, samples):
        shuffled_samples = random.sample(samples, len(samples))
        return shuffled_samples
        
        
    def format_samples(self, shuffled_samples):
        data = []
        labels = []
        for i in range(0, len(shuffled_samples)):
            data.append(shuffled_samples[i][0])
            labels.append(shuffled_samples[i][1])
        return data, labels
    #editing for 16 size
    def extract_samples_from_pdf(self, filepath, category):
        pdf_extractor = PDFExtractor()
        word_embedder = WordEmbedder(16, 64, 4, 1)   
        text = pdf_extractor.extract_text(filepath)
        embeddings = word_embedder.generate_embeddings(text)
        height = 16
        samples=[]
        for i in range(0, len(word_embedder.tokens)/height):
            pixel_data = []
            for j in range(0, height):
                pixel_data.append(embeddings[word_embedder.tokens[(i*height)+j]])
            #just saying 0 for now
            sample=(pixel_data, self.categories.index(category))
            samples.append(sample)
        #print samples
        data,labels = self.format_samples(samples)
        return data, labels
        
                           
        
        
    
        
            
            
            