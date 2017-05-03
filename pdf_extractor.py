import PyPDF2
import pattern
from pattern.en import parse
import string


class PDFExtractor():
    
    def read_pdf(self, filename):
        pdfFileObj = open(filename, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        if pdfReader.isEncrypted:
            pdfReader.decrypt('')        
        return pdfReader
    
    def concat_pages(self, pdfReader, pageRange):
        text = ""
        for i in range(pageRange[0], pageRange[1]):
            pageObj = pdfReader.getPage(i)
            text = text + " " + pageObj.extractText()    
        return text
    
    def extract_text(self, filename):
        pdfReader = self.read_pdf(filename)
        #edited for 10 pgs
        text = self.concat_pages(pdfReader, [0, pdfReader.getNumPages()])
        text = text.encode('utf8')
        tbl = string.maketrans('', '')
        text = text.translate(tbl, string.punctuation)
        #print text
        return text
    
