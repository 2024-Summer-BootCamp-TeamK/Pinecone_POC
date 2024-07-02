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

# 인덱스 이름 설정
index_name = "selective-time"

# 인덱스 열기
index = pc.Index(index_name)

# 모델 및 토크나이저 설정
model_name = "BM-K/KoSimCSE-roberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text_with_hf(text):
    #print("Embedding text with HuggingFace model...")
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
    #print(f"Generated embedding of shape: {embeddings.shape}")
    #print(f"Embedding values: {embeddings}")
    return embeddings.squeeze()

# 검색 함수 정의
def search_documents(query):
    #print(f"Searching documents for query: {query}")
    query_embedding = embed_text_with_hf(query)
    if query_embedding.shape != (768,):  # 임베딩 벡터의 크기 확인
        raise ValueError(f"Embedding size is {query_embedding.shape}, expected (768,)")
    #print(f"Query embedding generated: {query_embedding[:10]}...")  # 임베딩 벡터의 일부를 출력
    #print(f"Full query embedding: {query_embedding.tolist()}")
    result = index.query(vector=query_embedding.tolist(), top_k=5, include_metadata=True)
    #print("Pinecone query result:", result)
    return [match['metadata']['text'] for match in result['matches']]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    # 계약서 갖고오기
    path = "contracts/test1.txt"
    with open(path, "r", encoding="utf-8") as file:
        contract_text = file.read()
    # 사용자 질문 설정
    user_question = f"{contract_text}\n이것이 당신에게 주어지는 계약서 입니다\n"

    print(f"User question: {user_question}")

    # 문서 검색 결과 가져오기
    search_results = search_documents(user_question)
    context = " ".join(search_results)  # 검색된 문서 텍스트를 모두 하나의 문자열로 결합
    #print(f"Search results context: {context}")

    # OpenAI 모델 설정
    llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY)
    prompt_template = PromptTemplate(
        input_variables=["context", "user_question"],
        template="""
                주어진 Context를 법률 기준으로 삼고 Question으로 주어진 계약서 전체에서 주요 조항, 독소 조항, 모호한 표현을 모두 추출해 주세요.
                당신에게 주어진 계약서 내용의 원문도 출력해주세요
                당신은 변호사고 청중은 해당 계약서의 피계약자 입니다.
                피계약자는 20살 이상의 성인이나, 법률에 대한 지식이 계약자보다 상대적으로 적은 사람입니다.
                주요 조항은 계약이 성사될 시 피계약자가 가장 중요하게 살펴보아야 할 조항들을 의미합니다.
                독소 조항은 계약이 성사될 시 불리하게 작용할 수 있을 만한 조항이나 피계약자에게 잠재적 피해가 갈 수 있을 조항들을 모두 의미합니다.
                모호한 표현은 계약이 성사될 시, 추후에 피계약자에게 잠재적 피해가 끼칠 수 있을 만한 조항들을 의미합니다.
                추출한 내용 뒷부분에 정보의 출처를 알려주세요(ex: 00법 00조항)
                계약서의 맥락간 의미와 해당 법률 전체의 맥락을 파악하여 해당 법률에 어긋나거나, 피계약자에게 불리하게 작용할 수 있을 만한 조항을 나열하시오.
                그러나 계약서 상의 조건 및 조항에 대해서 참고한 법률과 비교하여 피계약자에게 불리한 점을 찾지 못했다면, 계약이 체결되어도 괜찮은 계약서라고 말해주세요.
                시간이 오래 걸려도 괜찮으니 당신이 가진 정보에 해당하는 모든 부분을 찾아주세요
                Context: {context}
                계약서: {user_question}
                Question: 
                1. 이 계약서의 주요 조항은 모두 무엇입니까?
                2. 이 계약서에서 독소 조항은 모두 무엇입니까?
                3. 이 계약서에서 모호한 표현은 모두 무엇입니까?
                맨 마지막 문단으로 (ex: 00법, 00법률, 00조)이런 식으로 독소조항에 대해 참고한 기존 법률과 비교하여, 어느 부분이 어떤 식으로 피계약자에게 불리하게 작용하는지 출력해주세요.
                또한, 이를 출력할 시 유사도가 높은 법 조항이 어디에 나와있는지도 출력하시오.

                독소조항에 대해서, 계약서에 제시되어 있는 조항을 법률에 근거하여 어떻게 수정하면 좋을 지에 대해서 수정안을 제시해주세요.
                출력 형식은 다음과 같습니다.
                origin - "기존 계약서 조항 중 독소 조항"
                recommend - "수정이 완료된 수정 조항"

                찾아낸 독소조항의 개수만큼 위와 같은 형식으로 출력해내시오.

                """
        )
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 질문과 검색된 문서 내용을 사용하여 모델에 invoke
    response = llm_chain.invoke({"context": context, "user_question": user_question})
    #print("Generated response:", response)

    # 결과 출력
    print("Answer:", response["text"])
