import os
from dataclasses import dataclass
from functools import cached_property

import fitz
import pandas as pd

from keys import API_KEY

os.environ['OPENAI_API_KEY'] = API_KEY

import logging

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

logging.basicConfig(level=logging.INFO)


# these are some of the available models. (using gpt-4 is very expensive)
MODELS = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']

def get_title(path):

    """returns the title of the document based on text size, position and page"""

    NON_TITLE_WORDS = ['draft', 'journal']

    pdf_details_list = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc.pages()):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks: 
                block_no = block['number']
                block_str = ''
                block_bold = []
                block_italic = []
                if 'lines' in block.keys():
                    for line in block['lines']:
                        for span in line['spans']:
                            block_str += span['text']
                            location_x = span['origin'][0]
                            location_y = span['origin'][1]
                            size = span['size']

                # add block to pdf_details if block has text
                if len(block_str.strip()) > 0:
                    pdf_details_list.append([page_no, block_no ,block_str, location_x, location_y, size])

    pdf_details = pd.DataFrame(pdf_details_list, columns=['page', 'block', 'text', 'location_x', 'location_y', 'size']).set_index(['page', 'block'])
    pdf_details['text'] = pdf_details['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii')) #remove weird encoding

    pdf_details = pdf_details.loc[pdf_details.text.str.strip() != '']
    for no_word in NON_TITLE_WORDS:
        pdf_details = pdf_details.loc[~pdf_details.text.str.lower().str.contains(no_word)]
    three_largest = pdf_details.loc[pdf_details['size'].isin(sorted(pdf_details['size'].unique())[-4:])]    
    title = three_largest.loc[0].sort_values('location_y')['text'].iat[0]
    if len(title) < 3:
        title = three_largest.iloc[0]['text']

    return title

def make_latex_document_str(header, data_description, main_text):
    main_text = main_text.replace("\n\n", " \\\\\n\\\\\n").replace('%', r'\%')
    data_description = data_description.replace("\n\n", " \\\\\n\\\\\n").replace('%', r'\%')
    string = (
        r"\documentclass{article}"+"\n"
        r"\title{"+header+"}"+"\n"
        r"\author{ChatGPT}"+"\n"
        r"\begin{document} "+"\n\n"
        r"\maketitle"+"\n"
        r"\section{Data}"+"\n"
        r""+data_description+" \n"
        r"\section{Summary}"+"\n"
        r""+main_text+" \n\n"
        r"\end{document}"
    )
    return string    

@dataclass
class BasePdfLLM:

    """
    BasePdfLLM is the Base Class to classes that use Langchain to:
    1) Load PDFs
    2) Generate Summaries 
    3) Asl the PDF questions 
    4) Create a .tex file that contains a summary + info about data 
    """

    filepath: str
    model_name: str = 'gpt-3.5-turbo-16k'
    temperature: float = 0

    def __post_init__(self):
        
        self.filepath = str(self.filepath)

        self.llm = ChatOpenAI(temperature=0, model_name=self.model_name)

        self.promt_template = (
            "Write a detailed summary of the text."
            "The summary contains one paragraph for each section of the text." 
            "Each paragraph contains up to 500 words and paragraphs are delimited with linebreaks" 
            "Section numbers are not mentioned in the detailed summary"
            "The detailed summary contains at most 3000 words." 
            "The detailed summary does not contain any graphs, charts, tables or exhibits."
            "The detailed summary does not contain the list of references."
            "The detailed summary does not contain the appendix."
            "Make sure that it is clear what is being summarised."
            "The detailed summary does not include statements without context."
            
            "The text is:\n"
            "{text}\n\n"
            "DETAILED SUMMARY:")
        
        self.refine_template = (
            "Your job is to produce a final summary\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary and create a refined summary if needed."
            "The refined summary contains one paragraph for each section of the text."
            "The refined summary does not contain graphs, charts, tables, exhibits, references and the appendix."
            "Each paragraph contains up to 500 words and paragraphs are delimited with linebreaks" 
            "The refined summary contains no more than 3000 words.\n" 
            
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary."
            "If the context isn't useful, return the original summary."
        )

        self.data_question = 'What data is used in the study, what countries are they using and what is the timeframe? '
        self.logger = logging.getLogger(f'{self.filepath}.log')

    def save_latex_summary(self, filename:str = None):

        title = get_title(self.filepath) + ' Summary'
        filename = filename if filename else f"{self.filepath.split('.')[0]}_{self.model_name}_summary.tex" 
        summary_text = self.detailed_summary
        data_description = self.ask_question(self.data_question)
        latex_str = make_latex_document_str(title, data_description, summary_text)
        with open(filename, 'w') as file:
            file.write(latex_str)

        self.logger.info(f'FILE: {filename} CREATED')
        
    def ask_question(self, question):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 50
        )
        splits = text_splitter.split_documents(self.pages)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        qa_chain = RetrievalQA.from_chain_type(self.llm,retriever=vectorstore.as_retriever())
        return qa_chain({"query": question})['result']
    
    @cached_property
    def detailed_summary(self):
        return self.get_summary(self.promt_template)

    @cached_property
    def pages(self):
        loader = PyPDFLoader(self.filepath)
        pages = loader.load()
        self.logger.info('LOADED PAGES')
        return pages 

    def get_summary(self, prompt_template):
        raise NotImplementedError('This is a Base Class')


class PdfLLMRefine(BasePdfLLM):

    """this uses the refine method to create a summary"""

    def get_summary(self, prompt_template):

        prompt = PromptTemplate.from_template(prompt_template)
        refine_prompt = PromptTemplate.from_template(self.refine_template)
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )
        self.logger.info('CREATED REFINE CHAIN')

        pages = self.pages
        self.logger.info('RUNNING REFINE CHAIN..')
        result = chain({"input_documents": pages}, return_only_outputs=True)
        self.logger.info('SUMMARY FINISHED')
        return result['output_text']


class PdfLLMStuff(BasePdfLLM):

    """This uses the stuff method to generate the summary"""
    
    def get_summary(self, prompt_template):
        
        #create chain
        prompt = PromptTemplate.from_template(prompt_template)        
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        self.logger.info('CREATED STUFF CHAIN')        
        
        #run chain
        pages = self.pages
        self.logger.info('RUNNING STUFF CHAIN..')
        result = stuff_chain.run(pages)
        self.logger.info('SUMMARY FINISHED')
        return result