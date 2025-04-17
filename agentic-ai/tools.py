import requests
from datetime import datetime
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.hanavector import HanaDB
from gen_ai_hub.proxy.langchain import init_llm
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from hdbcli import dbapi

HANA_HOST = '<your host address>'
HANA_USER = '<your user name>'
HANA_PASSWORD = '<your password>'

connection = dbapi.connect(
    HANA_HOST,
    port="443",
    user=HANA_USER,
    password=HANA_PASSWORD,
    autocommit=True,
    sslValidateCertificate=False,
)


def retriever(question: str):
    embedding_model = init_embedding_model('text-embedding-ada-002')
    llm = init_llm('gpt-4o-mini')
    
    
    prompt_template = """
    Use the following context to answer the question at the end. 
    If the answer is not directly stated, try your best based on the context. 
    Only say you don't know if the information is completely unavailable.

    {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    db = HanaDB(
        embedding=embedding_model,
        connection=connection,
        table_name="EMBEDDINGS_COLLECTION_DATA"
    )

    retriever = db.as_retriever(search_kwargs={'k': 10})
    qa = RetrievalQA.from_chain_type(
        llm= llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa.invoke({"query": question})
    return {"answer": response}


def get_time_now():
    """Returns the current local time as a formatted string."""
    return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]