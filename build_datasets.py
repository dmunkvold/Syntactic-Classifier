from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder
from training_set_builder import SampleBuilder

#didn't save to right folder, but otherwise worked! :)
#bob = SampleBuilder(['difficult','easy'])
#bob.convert_pdfs()

#???????? AUTHORS ?????????
bob = SampleBuilder(['Joyce','Haraway', 'Kant'])
bob.convert_pdfs()