from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# =========================================================
# Gemini Model
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

# =========================================================
# Prompt Template
# =========================================================

prompt = PromptTemplate(
    template="""
    how many male and female are there

    {data}
    """,
    input_variables=["data"]
)

# =========================================================
# Output Parser
# =========================================================

parser = StrOutputParser()

# =========================================================
# CSV Loader
# =========================================================
# Loads CSV file row-by-row as documents

loader = CSVLoader(
    file_path="Social_Network_Ads.csv",
    encoding="utf-8"
)

# =========================================================
# load() -> loads all rows into memory
# =========================================================

docs = loader.load()

print(f"Total rows loaded: {len(docs)}")

# Print first row
print("\nFirst Row:\n")
print(docs[0].page_content)

# =========================================================
# Combine rows into one text
# =========================================================

text = ""

for doc in docs:
    text += doc.page_content + "\n"

# =========================================================
# LangChain Pipeline
# =========================================================

chain = prompt | model | parser

# Generate response
result = chain.invoke({
    "data": text[:12000]
})



print("\nAI Analysis:\n")
print(result)