from pdf_extractor import PDFExtractor
from word_embedder import WordEmbedder

class TrainingSetBuilder():
    
    def __init__(self, categories):
        self.categories = categories
        
    def add_samples_from_pdf(self, pdfs, category):
        pdf_extractor = PDFExtractor()
        word_embedder = WordEmbedder()
        for pdf in pdfs:
            self.categories[category].extend(word_embedder.generate_embedded_samples(pdf_extractor.extract_text(pdf)))
            