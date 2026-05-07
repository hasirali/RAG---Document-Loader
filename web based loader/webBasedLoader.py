from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite"
)

# Prompt template
prompt = PromptTemplate(
    template="Write a summary for the following webpage:\n\n{text}",
    input_variables=["text"]
)

# Output parser
parser = StrOutputParser()

# =========================================================
# WebBaseLoader
# =========================================================
# Used to load content from websites/webpages.
# It extracts readable text from HTML pages.
url ="https://www.amazon.in/Samsung-Storage-MediaTek-Charging-Upgrades/dp/B0FN7QTRPY/ref=sr_1_1?_encoding=UTF8&s=electronics&sr=1-1"
loader = WebBaseLoader(url)

docs = loader.load()

print(f"Total documents loaded: {len(docs)}")

# Print first 500 characters
print(docs[0].page_content[:500])

# =========================================================
# Extract webpage text
# =========================================================

text = docs[0].page_content

# Create LangChain chain
chain = prompt | model | parser

# Generate summary
result = chain.invoke({
    "text": text[:12000]
})

# Print final summary
print("\nSummary:\n")
print(result)