import faiss
import json
import os
import numpy as np
from app.utils import embed_query, generate_reason
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔹 메타데이터 로드 (emotion_tags, vector 포함)
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# 🔹 LLM 세팅
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")


# 🔸 감정 카테고리 분류
def classify_emotion(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        다음 키워드는 꽃을 추천받기 위한 상황입니다:
        {keywords}

        이 키워드에서 느껴지는 중심 감정을 다음 중 하나로 분류해줘:

        사랑(고백), 사랑(부모), 사랑(영원),
        이별(분노), 이별(슬픔), 이별(화해),
        순수(응원), 순수(믿음),
        존경(우상),
        행복(기원), 행복(성공)

        가장 적절한 하나를 골라줘. 이유는 쓰지 말고 분류명만 줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"keywords": keywords}).strip()


# 🔸 키워드 확장
def expand_keywords(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        사용자가 입력한 키워드: {keywords}
        이 키워드를 바탕으로 감정과 상황을 포함한 자연스러운 문장으로 확장해줘.
        너무 길지 않고, 의도가 잘 드러나도록 말해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(keywords).strip()


# 🔹 추천 시스템 핵심 함수
def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    emotion_category = classify_emotion(keywords)
    query_vector = embed_query(expanded_query)

    # 🔍 감정 필터
    filtered_flowers = [
        flower for flower in metadata_list
        if "emotion_tags" in flower and emotion_category in flower["emotion_tags"]
    ]

    if not filtered_flowers:
        return {
            "expanded_query": expanded_query,
            "emotion_category": emotion_category,
            "recommendations": [],
            "error": f"'{emotion_category}' 감정에 해당하는 꽃이 없습니다."
        }

    # 🔧 임시 FAISS 인덱스 만들기
    dim = len(filtered_flowers[0]["vector"])
    tmp_index = faiss.IndexFlatL2(dim)
    vectors = np.array([flower["vector"] for flower in filtered_flowers]).astype("float32")
    tmp_index.add(vectors)

    distances, indices = tmp_index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        flower = filtered_flowers[idx]
        reason = generate_reason(expanded_query, flower["description"], flower["name"])
        results.append({
            "name": flower["name"],
            "description": flower["description"],
            "color": flower["color"],
            "season": flower["season"],
            "scent": flower["scent"],
            "reason": reason
        })

    return {
        "expanded_query": expanded_query,
        "emotion_category": emotion_category,
        "recommendations": results
    }
