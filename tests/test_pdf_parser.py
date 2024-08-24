import unittest
from indexify.extractors.pdf_parser import PDFParser
from indexify.extractor_sdk.utils import SampleExtractorData


class TestPdfParser(unittest.TestCase):

    def test_parse(self):
        pdf_file = SampleExtractorData().sample_invoice_pdf()
        pdf_parser = PDFParser(pdf_file.data)
        output = pdf_parser.parse()
        # Ensure that there is one page here 
        self.assertEqual(len(output), 1)
        # Ensure that the first page has a text and a table
        print(output)
        self.assertEqual(len(output[0].fragments), 6)


if __name__ == '__main__':
    unittest.main()