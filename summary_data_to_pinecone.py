import os
import json
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer
import pinecone
import hashlib
import numpy as np

# .env 파일에서 환경 변수 로드
load_dotenv()

# Pinecone API 키 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone 초기화
pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')

# 인덱스 이름과 설정
index_name = "legal-docs"
dimension = 768  # HuggingFace KoSimCSE 로버타 모델의 임베딩 차원
metric = "cosine"

# 인덱스 존재 여부 확인 및 생성
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        pod_type="p1"
    )

index = pinecone.Index(index_name)

class MyEmbeddingModel:  # 커스텀 임베딩 클래스
    def __init__(self, model_name):  # 클래스 생성자
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, doc):  # 문서 임베딩 메서드
        inputs = self.tokenizer(doc, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def embed_query(self, text):  # 쿼리 임베딩 메서드
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

embed_model_name = "BM-K/KoSimCSE-roberta-multitask"
embedding_model = MyEmbeddingModel(embed_model_name)

def generate_vector_id(text, index):
    unique_string = f"{text}_{index}"
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

def insert_data_to_pinecone(data):
    vectors = []
    for item in data:
        for i, detail in enumerate(item.get('세부내용', [])):
            text = detail['내용'] if isinstance(detail['내용'], str) else ' '.join(detail['내용'])
            inputs = embedding_model.tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            vector_id = generate_vector_id(text, i)
            vectors.append({
                'id': vector_id,
                'values': embeddings,
                'metadata': {
                    '주제': item['주제'],
                    '항목': item['항목'],
                    '설명': detail['설명'],
                    '내용': detail['내용']
                }
            })

    # Pinecone에 벡터 삽입
    index.upsert(vectors)

if __name__ == "__main__":
    # 병합된 JSON 파일 경로
    json_file_path = 'datas/summary/건설일용근로자_생활법령.json'

    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 데이터 Pinecone에 삽입
    insert_data_to_pinecone(data)

    print("JSON data embedded and uploaded to Pinecone index.")
