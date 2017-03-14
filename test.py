from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder
from PIL import Image

def showImg(pixeldata):
    img = Image.fromarray(pixeldata, 'LA')
    img.save('my.png')
    img.show()
    
ex = PDFExtractor()
text = ex.extract_text("/Users/David/Downloads/Charlotte's Web.pdf")
emb = WordEmbedder(100)
emb.generate_embedded_samples(text)

#for i in range(0, 10):
#    showImg(img_samples[i])