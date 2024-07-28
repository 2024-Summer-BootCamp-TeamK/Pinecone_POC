import os

import fitz
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataToPinecone import pc
import prompts


# .env 파일에서 환경 변수 로드
load_dotenv()

# Pinecone API 키와 OpenAI API 키 로드
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 인덱스 이름 설정
index_names = ["legal-docs", "lawbot"]

# 모델 및 토크나이저 설정
# model_name = "BM-K/KoSimCSE-roberta-multitask"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)


tokenizer = AutoTokenizer.from_pretrained("hyunseoki/ko-en-llama2-13b")
model = AutoModelForCausalLM.from_pretrained("hyunseoki/ko-en-llama2-13b")

def embed_text_with_hf(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
    return embeddings.squeeze()

# 검색 함수 정의
def search_documents(index, query):
    query_embedding = embed_text_with_hf(query)
    if query_embedding.shape != (768,):  # 임베딩 벡터의 크기 확인
        raise ValueError(f"Embedding size is {query_embedding.shape}, expected (768,)")
    result = index.query(vector=query_embedding.tolist(), top_k=6, include_metadata=True)
    return [match['metadata']['text'] for match in result['matches']]

def search_documents_legal_docs(index, query):
    # 사용자 질문을 임베딩 벡터로 변환하기
    query_embedding = embed_text_with_hf(query)
    if query_embedding.shape != (768,):  # 임베딩 벡터의 크기 확인
        raise ValueError(f"Embedding size is {query_embedding.shape}, expected (768,)")
    result = index.query(vector=query_embedding.tolist(), top_k=6, include_metadata=True)
    return [match['metadata']['세부항목'] for match in result['matches']]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


if __name__ == "__main__":
    # 계약서 갖고오기
    path = "wrong_contract_construct/건설일용근로자 표준근로계약서 .pdf"
    contract_text = extract_text_from_pdf(path)
    # 사용자 질문 설정
    user_question = f"{contract_text}\n이 법률적으로 검토해야 할 계약서 입니다\n"

    refined_search_results = []
    for index_name in index_names:
        try:
            index = pc.Index(index_name)
            if index_name == "legal-docs":
                refined_search_results.extend(search_documents_legal_docs(index, user_question))
            else:
                refined_search_results.extend(search_documents(index, user_question))
        except Exception as e:
            print(f"Error searching in index {index_name}: {str(e)}")

    # 검색된 문서 텍스트를 모두 하나의 문자열로 결합
    context = " ".join(refined_search_results)
    # OpenAI 모델 설정
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["context", "user_question"],
        template=prompts.GUIDELINE_PROMPT
        )
    llm_sequence = prompt_template | llm

    # 질문과 검색된 문서 내용을 사용하여 모델에 invoke
    response = llm_sequence.invoke({"context": context, "user_question": user_question})
    raw_result = response.content
    # 결과 출력
    print("Answer : ", raw_result)

    # parsed_result = json.loads(raw_result)
    # print(parsed_result)
    # articles = []
    # for i in range(len(parsed_result)):
    #     article = {
    #         "sentence": parsed_result[i]["sentence"],
    #         "description": parsed_result[i]["description"],
    #         "law": parsed_result[i]["law"],
    #     }
    #     articles.append(article)
    # print(articles)