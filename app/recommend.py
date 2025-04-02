import faiss
import json
import os
import numpy as np
from typing import List
from app.utils import embed_query, generate_reason
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔹 FAISS 인덱스 및 메타데이터 로딩
index = faiss.read_index("flower_index.faiss")
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# 🔹 LangChain LLM 설정
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# 🔹 키워드 리스트 → 하나의 확장 문장 생성
def expand_keywords(keywords: List[str]) -> str:
    keywords_str = ", ".join(keywords)
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        사용자가 입력한 키워드들: {keywords}
        이 키워드들을 바탕으로 감정과 상황이 느껴지는 자연스러운 하나의 문장 또는 단락으로 확장해줘.
        키워드들이 연결된 이야기처럼 이어지게 하고, 줄거리처럼 부드럽게 써줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(keywords=keywords_str).strip()

# 🔹 꽃 추천 함수
def get_flower_recommendations(keywords: List[str], top_k: int = 3):
    print("📥 [시작] get_flower_recommendations()")
    print("🔤 입력 키워드 리스트:", keywords)

    # 1. 키워드 확장
    expanded_query = expand_keywords(keywords)
    print("🪄 확장된 문장:", expanded_query)

    # 2. 임베딩 → (1, D) numpy 배열
    try:
        raw_vector = embed_query(expanded_query)
        query_vector = np.array(raw_vector).reshape(1, -1)
    except Exception as e:
        print("❌ 임베딩 변환 에러:", e)
        return {"error": f"임베딩 에러: {e}"}

    # 3. FAISS 검색
    try:
        distances, indices = index.search(query_vector, 10)
        print("🔎 FAISS 결과:", indices)
    except Exception as e:
        print("❌ FAISS 검색 에러:", e)
        return {"error": f"faiss.search 에러: {e}"}

    # 4. 추천 결과 정리
    results = []
    seen = set()
    for idx in indices[0]:
        flower = metadata_list[idx]
        if flower["name"] in seen:
            continue
        seen.add(flower["name"])

        reason = generate_reason(expanded_query, flower["description"], flower["name"])
        results.append({
            "name": flower["name"],
            "description": flower["description"],
            "color": flower["color"],
            "season": flower["season"],
            "scent": flower["scent"],
            "reason": reason
        })

        if len(results) == top_k:
            break

    return {
        "expanded_query": expanded_query,
        "recommendations": results
    }
