from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder
from image_generator import ImageGenerator
from PIL import Image

def showImg(pixeldata):
    img = Image.fromarray(pixeldata, 'LA')
    img.save('my.png')
    img.show()
    
ex = PDFExtractor()
text = ex.extract_text("./lib/pdfs/easy/Charlotte's Web.pdf")
emb = WordEmbedder(64, 64, 2, 1)
embeddings = emb.generate_embeddings(text)
imggen = ImageGenerator(emb.tokens, embeddings)
imggen.generate_images(64)
#for i in range(0, 10):
#    showImg(img_samples[i])