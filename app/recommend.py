import os
import json
import faiss
import numpy as np
from app.utils import embed_query, generate_reason
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 패스 전체 열람
INDEX_PATH = "flower_index.faiss"
SEARCH_EXPANSION_FACTOR = 5
index = faiss.read_index(INDEX_PATH)

# flower_metadata 로드
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# 감정 정의 prompt
emotion_prompt = PromptTemplate(
    input_variables=["keywords"],
    template="""
    다음 키워드는 꽃을 추천받기 위한 구체적 상황입니다:
    {keywords}

    다음 감정 카테고리 중 가장 적절한 하나만 고르어주세요:
    사랑(강렬한), 사랑(순수한), 사랑(영원한), 사랑(행복한), 사랑(따뜻한),
    슬픔(화해), 슬픔(이별), 슬픔(그리움), 슬픔(위로),
    축하(승진), 축하(개업), 축하(합격), 축하(생일), 축하(출산),
    응원(새로운 시작), 응원(합격 기원), 응원(격려), 응원(꿈을 향한 도전),
    행복(영원한), 행복(순수한), 행복(함께한), 행복(다가올),
    특별함(비밀), 특별함(신비), 특별함(마법), 특별한(고귀), 특별한(고급)
    """
)
emotion_chain = emotion_prompt | llm

expand_prompt = PromptTemplate(
    input_variables=["base_sentence"],
    template="""
    바탕 문장을 감정적으로 4~6문장으로 확장해주세요.
    자세하고 재료가 감독되도록 만드드릴까요.
    
    문장: {base_sentence}
    """
)
expand_chain = expand_prompt | llm


def classify_emotion(keywords: str) -> str:
    return emotion_chain.invoke({"keywords": keywords}).content.strip()


def expand_keywords(keywords: list[str], structured: bool = True) -> str:
    if structured and isinstance(keywords, list) and len(keywords) >= 5:
        target = keywords[0]
        gender = keywords[1]
        emotion_main = keywords[2]
        emotion_detail = keywords[3]
        personality = keywords[4]

        base = (
            f"나는 성별이 {gender}인 {target}에게 {emotion_main}의 감정에 {emotion_detail}을 더해서 전하고 싶어요. "
            f"그 사람은 {personality}, 가장 사랑할 만한 감정을 가진 방식으로 전해야 해요."
        )
        return expand_chain.invoke({"base_sentence": base}).content.strip()

    raise ValueError("키워드는 그대, 성별, 감정, 세부 감정, 성향 포함 5개 이상이어야 합니다.")


def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    emotion_category = classify_emotion(keywords)
    query_vector = embed_query(expanded_query)

    # 감정 카테고리가 맞는 꽃을 무엇보다 먼저 검색
    filtered_indices = [i for i, flower in enumerate(metadata_list) if emotion_category in flower.get("emotion_tags", [])]

    if not filtered_indices:
        filtered_indices = list(range(len(metadata_list)))  # fallback

    # FAISS가 가장 가게적으로 검색
    distances, indices = index.search(np.array(query_vector).astype("float32"), top_k * SEARCH_EXPANSION_FACTOR)

    results_with_score = []
    for i in indices[0]:
        if i not in filtered_indices:
            continue
        flower = metadata_list[i]
        base_score = distances[0][list(indices[0]).index(i)]
        results_with_score.append((i, base_score))

    results_with_score.sort(key=lambda x: x[1])
    seen_names = set()
    final_results = []

    for i, _ in results_with_score:
        flower = metadata_list[i]
        if flower["name"] in seen_names:
            continue
        seen_names.add(flower["name"])

        try:
            reason = generate_reason(expanded_query, flower["description"], flower["name"])
        except Exception:
            reason = "[\ucd94\ucc9c \uc774\uc720 \uc0dd\uc131 \uc2e4\ud328]"

        final_results.append({
            "FLW_IDX": flower["FLW_IDX"],
            "reason": reason
        })

        if len(final_results) >= top_k:
            break

    return {"recommendations": final_results}
