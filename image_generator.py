from PIL import Image
import numpy
from pylab import *


class ImageGenerator():
    
    def __init__(self, tokens, embeddings):
        self.tokens = tokens
        self.embeddings = embeddings
        self.samples = []
    
    def generate_images(self, sample_length):
    #def generate_images(self, sample_length, folder, tag):
        height = sample_length
        print self.embeddings
        for i in range(0, len(self.tokens)/height):
            pixel_data = []
            for j in range(0, height-1):
                pixel_data.append(self.embeddings[self.tokens[(i*height)+j]])
            norm = self.normalize(pixel_data)
            print 'norm!',norm
            #potential to speed up process of pixel data to training set by avoiding saving it as image. Right
            # now I am going to save as images and then grab the data from the images 
            
            #self.samples.append(pixel_data)
            img = Image.fromarray(norm)
            #img.save(folder + '/img'+ str(i) + '_' + tag + '.png')
            img.save('./immmg'+ str(i) + '.png')
        #return self.samples
            # create a new figure
            #figure()
            #gray()
            # show contours with origin upper left corner
            #contour(img, origin='image')
            #axis('equal')
            #axis('off')            
            #figure()


            #hist(array(img).flatten(), 128)

            #show()  
    #add variables to nrom pd size
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
            