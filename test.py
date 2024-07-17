import os
import json
from dotenv import load_dotenv
import torch
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModel, AutoTokenizer
import hashlib

# .env 파일에서 환경 변수 로드
load_dotenv()

# Pinecone API 키 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# 이름과 설정
index_name = "legal-docs"
dimension = 768  # HuggingFace KoSimCSE 로버타 모델의 임베딩 차원
metric = "cosine"

# 인덱스 존재 여부 확인 및 생성
if index_name not in [index.name for index in pc.list_indexes().indexes]:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

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

def extract_text(detail):
    combined_text = []

    def flatten_text(data):
        if isinstance(data, str):
            combined_text.append(data)
        elif isinstance(data, list):
            for item in data:
                flatten_text(item)
        elif isinstance(data, dict):
            for value in data.values():
                flatten_text(value)

    flatten_text(detail)
    return ' '.join(combined_text)

def truncate_text(text, max_length):
    tokens = embedding_model.tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return embedding_model.tokenizer.convert_tokens_to_string(tokens)

if __name__ == "__main__":
    # 병합된 JSON 파일 경로
    json_file_path = 'datas/summary/건설일용근로자_생활법령.json'

    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    vectors = []
    for item in data:
        for i, detail in enumerate(item.get('세부항목', [])):
            combined_text = extract_text(detail)
            if combined_text:
                truncated_text = truncate_text(combined_text, embedding_model.tokenizer.model_max_length)
                inputs = embedding_model.tokenizer(truncated_text, return_tensors='pt', max_length=512, truncation=True)
                with torch.no_grad():
                    embeddings = embedding_model.model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
                vector_id = generate_vector_id(combined_text, i)
                vectors.append({
                    'id': vector_id,
                    'values': embeddings,
                    'metadata': {
                        '주제': item.get('주제', ''),
                        '항목': item.get('항목', ''),
                        '세부항목': combined_text
                    }
                })

    # Pinecone에 벡터 업로드
    index.upsert(vectors)

    print("Vectors successfully uploaded to Pinecone.")
