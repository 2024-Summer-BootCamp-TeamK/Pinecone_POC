import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import torch
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
import hashlib
import numpy as np

# .env 파일에서 환경 변수 로드
load_dotenv()

# Pinecone API 키 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone 초기화
pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 이름과 설정
index_name = "selective-time"
dimension = 768  # HuggingFace KoSimCSE 로버타 모델의 임베딩 차원
metric = "cosine"

# 인덱스 존재 여부 확인 및 생성
existing_indexes = [index.name for index in pc.list_indexes().indexes]
if index_name not in existing_indexes:
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


# PDF 로드 및 전처리 함수 정의
def process_pdf(path):
    # PDF 로드 및 텍스트 추출
    loader = PyPDFLoader(path)
    docs = loader.load_and_split()
    print(f"문서의 수: {len(docs)}")

    def create_metadata(docs):
        for idx, doc in enumerate(docs):
            doc.metadata["category"] = "2024.06"
            doc.metadata["path"] = "datas"

    create_metadata(docs)

    # 텍스트 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "."],
        length_function=len,
    )

    documents = text_splitter.split_documents(docs)
    print(f"Number of documents after splitting: {len(documents)}")

    return documents


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


if __name__ == "__main__":
    # PDF 파일 경로 설정
    path = "selectiveTime/장애인고용촉진및직업재활법.pdf"

    documents = process_pdf(path)

    data = []
    for i, doc in enumerate(documents):
        # 텍스트 조각 임베딩
        embedding = embedding_model.embed_documents(doc.page_content)

        # 벡터 ID 생성
        vector_id = generate_vector_id(doc.page_content, i)

        # 데이터 생성
        data.append({
            "id": vector_id,
            "values": np.array(embedding, dtype=np.float32).mean(axis=0).tolist(),  # numpy 배열을 리스트로 변환
            "metadata": {"source": path, "text": doc.page_content}
        })

    # 데이터 업로드
    index.upsert(vectors=data)

    print("PDF text embedded and uploaded to Pinecone index.")
