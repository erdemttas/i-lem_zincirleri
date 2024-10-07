from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from langchain.chains.openai_functions import create_openai_fn_runnable

import os
from dotenv import load_dotenv

load_dotenv()

my_key_openai = os.getenv("openai_key")

class Insan(BaseModel):
    """Bir insan hakkında tanımlayıcı bilgiler"""
    isim: str = Field(..., description="Kişinin ismi")
    yas: int = Field(..., description="Kişinin yaşı")
    meslek: Optional[str] = Field(None, description="Kişinin mesleği")


class Sehir(BaseModel):
    """Bir şehir hakkında tanımlayıcı bilgiler"""
    isim: str = Field(..., description="Şehrin ismi")
    plaka_no: int = Field(..., description="Şehrin plaka numarası")


llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=my_key_openai)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Sen varlıkları kaydetme konusunda dünyanın en başarılı algoritmasısın"),
        ("human", "Şu verdiğim girdideki varlıkları kaydetmek için gerekli fonksiyonlara çağrı yap {input}"),
        ("human", "İpucu: Doğru formatta yanıtladığından emin ol")
    ]
)

chain_2 = create_openai_fn_runnable([Insan, Sehir], llm, prompt)

print(chain_2.invoke({"input": "Aydın 34 yaşında, başarılı bir bilgisayar mühendisi"}))
print(chain_2.invoke({"input": "Aydın'da hava her zaman sıcaktır ve bu yüzden 09 plakalı araçlarda klima sürekli çalışır."}))

my_parameter = chain_2.invoke({"input": "Aydın 34 yaşında, başarılı bir bilgisayar mühendisiydi"})

# Gelen dict yapısını Insan modeline dönüştür
insan_objesi = my_parameter

def Insan_Kaydet(insan: Insan):
    print(f"İsim: {insan.isim}, Yaş: {insan.yas}, Meslek: {insan.meslek}")

Insan_Kaydet(insan_objesi)
