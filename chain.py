# Create Stuff Documents Chain
from langchain.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


import os
from dotenv import load_dotenv

load_dotenv()

my_key_openai = os.getenv("openai_key")
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=my_key_openai)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Burada ismin geçen kişilerin en sevdiği rengi tek tek yaz:\n\n{context}")
    ]
)

docs = [
    Document(page_content="Gamze kırmızıyı sever ama sarıyı sevmez"),
    Document(page_content="Murat yeşili sever ama maviyi sevdiği kadar değil"),
    Document(page_content="Burak'a sorsan favori rengim yok der ama belli ki turuncu rengi seviyor")
]

chain_1 = create_stuff_documents_chain(llm, prompt)

print(chain_1.invoke({"context": docs}))








