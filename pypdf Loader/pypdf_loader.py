from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template="write a summary for following text {text}",
    input_variables=['text']
)
parser = StrOutputParser()

loader = PyPDFLoader('dl-curriculum.pdf')
docs = loader.load()

# print(len(docs)) #output 23 as 23 pages
# print(docs[0].page_content)
# print(docs[1].metadata) #Output{'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': 'dl-curriculum.pdf', 'total_pages': 23, 'page': 1, 'page_label': '2'}
chain = prompt | model | parser

# Run chain
result = chain.invoke({"text": docs[0].page_content})

# Print summary
print(result)

