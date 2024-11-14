# custom function for text cleaning
def clean_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove newline characters
    text = text.replace('\n', ' ').replace('\r', ' ')

    return text

# Custom function to load PDFs using PyMuPDF
class PDFLoader:
    def __init__(self, file_path,page_start,page_end):
        self.file_path = file_path
        self.page_start = page_start
        self.page_end = page_end

    def load(self):
        documents = []
        # Open the PDF file
        with fitz.open(self.file_path) as pdf_document:
            # Iterate through each page
            for page_num in range(self.page_start,self.page_end):
                page = pdf_document[page_num]
                text = page.get_text("text")
                text = clean_text(text)
                # Create a LangChain Document for each page
                if text:
                    documents.append(Document(page_content=text, metadata={"page_number": page_num + 1}))
        return documents

def load_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def load_and_split_documents(file_path, start_page, end_page, chunk_size=500, chunk_overlap=50):
    pdf_loader = PDFLoader(file_path, start_page, end_page)
    documents = pdf_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=[". ", "? ", "! ", " "])
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def create_vector_store_retriever(documents, embeddings):
    embeddings  = load_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    return retriever

def load_language_model(repo_id, api_token, temperature=0.7):
    return HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": temperature,"max_length":10000},
        huggingfacehub_api_token=api_token
    )




def create_rag_chain(retriever):


  system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.Just give the final answer. Remember you do not have to extent the input query. Just answer it."
    "\n\n"
    "{context}"
)
  prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")

    ]
)




  question_answer_chain = create_stuff_documents_chain(llm, prompt)
  rag_chain = create_retrieval_chain(retriever, question_answer_chain)

  return rag_chain




def generate_response(user_query,rag_chain):
  results = rag_chain.invoke({"input": user_query.lower()})
  answer = results['answer'].replace("\n",'')
  return answer

