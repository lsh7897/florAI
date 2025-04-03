import faiss
import json
import os
import numpy as np
from app.utils import embed_query, generate_reason
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔹 FAISS 인덱스 로드 (벡터만 저장)
index = faiss.read_index("flower_index.faiss")

# Load metadata
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# LLM setup
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

def classify_emotion(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        다음 키워드는 꽃을 추천받기 위한 상황입니다:
        {keywords}

        다음 감정 카테고리 중 가장 적절한 하나만 고르어줘 (목록 외 감정은 절대 사용하지 마):

        사랑(강렬한), 사랑(순수한), 사랑(영원한), 사랑(행복한), 사랑(따뜻한),
        슬픔(화해), 슬픔(이별), 슬픔(그리움), 슬픔(위로),
        축하(승진), 축하(개업), 축하(합격), 축하(생일), 축하(출산),
        응원(새로운 시작), 응원(합격 기원), 응원(격려), 응원(꿈을 향한 도전),
        행복(영원한), 행복(순수한), 행복(함께한), 행복(다가올),
        특별함(비밀), 특별함(신비), 특별함(마법), 특별한(고귀), 특별한(고급)

        정확히 위 목록 중 하나만 출력해줘. 이유는 쓰지 마.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"keywords": keywords}).strip()

def expand_keywords(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        사용자가 입력한 키워드: {keywords}
        이 키워드를 바탕으로 감정과 상황을 포함한 자연스럽고 모든 의도가 잘 전달되게 문장으로 확장해줘.
        너무 길지 않고, 말하고자 하는 목적이 잘 나타나게게 해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(keywords).strip()

def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    emotion_category = classify_emotion(keywords)
    query_vector = embed_query(expanded_query)

    distances, indices = index.search(np.array(query_vector).astype("float32"), top_k * 5)

    results_with_score = []
    for i in indices[0]:
        flower = metadata_list[i]
        base_score = distances[0][list(indices[0]).index(i)]
        boost = -0.2 if emotion_category in flower.get("emotion_tags", []) else 0.0
        final_score = base_score + boost
        results_with_score.append((i, final_score))

    results_with_score.sort(key=lambda x: x[1])
    top_indices = [i for i, _ in results_with_score[:top_k]]

    results = []
    for idx in top_indices:
        flower = metadata_list[idx]
        reason = generate_reason(expanded_query, flower["description"], flower["name"])
        results.append({
            "FLW_IDX": flower["FLW_IDX"],
            "reason": reason
        })

    return {
        "recommendations": results
    }