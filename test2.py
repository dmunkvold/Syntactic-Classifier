from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder
from training_set_builder import SampleBuilder

#didn't save to right folder, but otherwise worked! :)
#bob = SampleBuilder(['difficult','easy'])
#bob.convert_pdfs()

#bob.convert_pdfs({'difficult': ["./lib/pdfs/difficult/The-Metamorphosis.pdf"]})
#bob.convert_pdfs({'difficult': ["./lib/pdfs/difficult/Ulysses.pdf", "./lib/pdfs/difficult/The-Metamorphosis.pdf"], 'easy':["./lib/pdfs/easy/Charlotte's Web.pdf"]})
#bob.convert_pdfs({'easy': []})
#???????? AUTHORS ?????????
bob = SampleBuilder(['Joyce','Haraway', 'Kant'])
bob.convert_pdfs()