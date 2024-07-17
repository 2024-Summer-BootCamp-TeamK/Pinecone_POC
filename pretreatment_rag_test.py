import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from dataToPinecone import pc

# .env 파일에서 환경 변수 로드
load_dotenv()

# Pinecone API 키와 OpenAI API 키 로드
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 인덱스 이름 설정
index_names = ["legal-docs", "lawbot"]

# 모델 및 토크나이저 설정
model_name = "BM-K/KoSimCSE-roberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text_with_hf(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
    return embeddings.squeeze()

# 검색 함수 정의
def search_documents(index, query):
    # 사용자 질문을 임베딩 벡터로 변환하기
    query_embedding = embed_text_with_hf(query)
    if query_embedding.shape != (768,):  # 임베딩 벡터의 크기 확인
        raise ValueError(f"Embedding size is {query_embedding.shape}, expected (768,)")
    result = index.query(vector=query_embedding.tolist(), top_k=7, include_metadata=True)
    return [match['metadata']['text'] for match in result['matches']]

def search_documents_legal_docs(index, query):
    # 사용자 질문을 임베딩 벡터로 변환하기
    query_embedding = embed_text_with_hf(query)
    if query_embedding.shape != (768,):  # 임베딩 벡터의 크기 확인
        raise ValueError(f"Embedding size is {query_embedding.shape}, expected (768,)")
    result = index.query(vector=query_embedding.tolist(), top_k=7, include_metadata=True)
    return [match['metadata']['세부항목'] for match in result['matches']]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":

    # 사용자 질문 설정
    user_question = " '근무 시간: 주 7일, 일 12시간 (오전 8시부터 오후 8시까지, 휴게 시간 포함)'에 대해서 잘못된 법적 근거를 설명하시오 "

    print(f"User question: {user_question}")

    # Step 1: Search in "legal-docs" Index
    index = pc.Index("legal-docs")
    initial_search_results = search_documents_legal_docs(index, user_question)

    for i, doc in enumerate(initial_search_results, 1):
        print(f"initial Search result {i}: {doc}")

    # Combine the initial search results as a new context for the refined search
    combined_context = " ".join(initial_search_results)

    # Step 2: Refine Search in "lawbot" Index using the combined context
    refined_search_results = []
    index = pc.Index("lawbot")
    refined_search_results.extend(search_documents(index, combined_context))

    # 검색된 문서 출력
    for i, doc in enumerate(refined_search_results, 1):
        print(f"refined Search result {i}: {doc}")

    context = " ".join(refined_search_results)  # 검색된 문서 텍스트를 모두 하나의 문자열로 결합
    print(f"Context: {context}")

    # OpenAI 모델 설정
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["context", "user_question"],
        template="""
                You are an AI trained on legal documents. Provide answers based on the given context.
                Context: {context}
                Question: {user_question}
                Answer: 
                """
        )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 질문과 검색된 문서 내용을 사용하여 모델에 invoke
    response = llm_chain.invoke({"context": context, "user_question": user_question})
    print("Generated response:", response["text"])

