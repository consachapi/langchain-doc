from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from upload_data import upload_docs, create_vectorstore


AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"


def init(path_file):
    llm = Ollama(model="llama3")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embed_model, persist_directory="chroma_db_dir", collection_name="stanford_report_data")
    total_rows = len(vectorstore.get()['ids'])
    
    if total_rows == 0:
        docs = upload_docs(path_file)
        vectorstore = create_vectorstore(docs)
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    custom_prompt_template = """Usa la siguiente información para responder la pregunta del usuario.
Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

    Contexto: {context}
    Respuesta: {question}

    Solo devuelve la respuesta útil a continuación y nada más que en español.
    Respuesta util:
    """

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("¡Bienvenido al chat! Escribe 'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break

        respuesta = qa.invoke({"query": pregunta})
        metadata = []
        for _ in respuesta['source_documents']:
            metadata.append(('page: '+str(_.metadata['page']), _.metadata['file_path']))
        print(f"{VERDE}Asistente:{RESET}", respuesta['result'], '\n', metadata)

if __name__ == "__main__":
    path_file = "proteccion-datos-personales.pdf"
    init(path_file)