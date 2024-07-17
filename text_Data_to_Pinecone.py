import hashlib
import os
import re

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)

# 인덱스 이름과 설정
index_name = "lawbot"
dimension = 768  # HuggingFace KoSimCSE 로버타 모델의 임베딩 차원
metric = "cosine"
# 텍스트 길이에 영향을 받지 않는 cosine 알고리즘 채택, 텍스트의 방향성(내용)을 잘 반영해줌.

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

# ( 전체 파일)
# 텍스트 파일 로드 및 전처리 함수 정의
# def process_text_files(directory):
#     all_documents = []
#
#     # 디렉토리 내의 모든 텍스트 파일 읽기
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".txt"):
#                 file_path = os.path.join(root, file)
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     text = f.read()
#
#                 # 텍스트 분할 설정
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1200,
#                     chunk_overlap=100,
#                     separators=["\n\n", "\n", "."],
#                     length_function=len,
#                 )
#
#                 documents = text_splitter.split_text(text)
#                 for doc in documents:
#                     all_documents.append({
#                         "text": doc,
#                         "source": file_path
#                     })
#     return all_documents

def process_text_file(file_path):
    # 텍스트 파일 경로 설정
    all_documents = []
    file_name = os.path.basename(file_path).replace('.txt', '')

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

        # Step 1: '제0장' 단위로 텍스트 나누기
    chapters = re.split(r'(\n*제\d+장의?\d* [^\n]+)', text)
    print(chapters)
    if len(chapters) > 1:
        chapters = [chapters[i] + chapters[i + 1] for i in range(1, len(chapters), 2)]
    else:
        chapters = [text]
    chunks = []

    # Step 2: 각 '제0장' 단위로 나눠진 텍스트를 '제0조' 단위로 나누기
    for chapter in chapters:
        chapter_title_match = re.match(r'(\n*제\d+장의?\d* [^\n]+)', chapter)
        chapter_title = chapter_title_match.group(1) if chapter_title_match else ""
        sections = re.split(r'(\n*제\d+조의?\d*\([^)]+\))', chapter)
        # 결과 리스트를 생성
        results = []
        for i in range(1, len(sections), 2):
            title = sections[i].strip()
            content = sections[i + 1]
            results.append(title)
            results.append(content.strip())

        # 각 섹션을 청크로 분할하고 파일 이름과 장 제목을 앞에 추가
        previous_section_title = ""
        for i in range(0, len(results), 2):
            title = results[i]
            content = results[i + 1]
            combined_text = f"{title} {content}"

            section_chunks = split_section(combined_text)
            # print(f"section_chunks...: {section_chunks}")

            # 이전 섹션 제목을 추가하여 연속성 보장
            for j, chunk in enumerate(section_chunks):
                # print(f"연속성{j}: {chunk}")
                if j == 0:
                    chunk = f"{file_name} {chapter_title} {chunk}"
                else:
                    chunk = f"{file_name} {chapter_title} {title} {chunk}"
                chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")

    for doc in chunks:
        all_documents.append({
            "text": doc,
            "source": file_path
        })

    return(all_documents)

def split_section(section, chunk_size=1200, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "."],
        length_function=len,
    )
    return text_splitter.split_text(section)


# 커스텀 임베딩 클래스 정의
class MyEmbeddingModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, doc):
        inputs = self.tokenizer(doc, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def embed_query(self, text):
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
    # 텍스트 파일 경로 설정
    directory = "datas/change_text/임금채권보장법.txt"
    all_documents = process_text_file(directory)

    data = []
    for i, doc in enumerate(all_documents):
        # 텍스트 조각 임베딩
        embedding = embedding_model.embed_documents(doc["text"])

        # 벡터 ID 생성
        vector_id = generate_vector_id(doc["text"], i)

        # 데이터 생성
        data.append({
            "id": vector_id,
            "values": np.array(embedding, dtype=np.float32).mean(axis=0).tolist(),  # numpy 배열을 리스트로 변환
            "metadata": {"source": doc["source"], "text": doc["text"]}
        })

    # 데이터 업로드
    index.upsert(vectors=data)

    print("Text files embedded and uploaded to Pinecone index.")