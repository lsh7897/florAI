import os
import json
import faiss
import numpy as np
from app.utils import embed_query, generate_reason
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔹 FAISS index 경로
INDEX_PATH = "faiss_index/flower_index.faiss"
SEARCH_EXPANSION_FACTOR = 5  # top_k * 5 검색

# 🔹 Load FAISS index
index = faiss.read_index(INDEX_PATH)

# 🔹 Load flower metadata
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# 🔹 Set up shared LLM
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# 🔹 Prompt & Chain (Emotion Classification)
emotion_prompt = PromptTemplate(
    input_variables=["keywords"],
    template="""
    다음 키워드는 꽃을 추천받기 위한 상황입니다:
    {keywords}

    다음 감정 카테고리 중 가장 적절한 하나만 골라줘 (정확히 하나만):
    사랑(강렬한), 사랑(순수한), 사랑(영원한), 사랑(행복한), 사랑(따뜻한),
    슬픔(화해), 슬픔(이별), 슬픔(그리움), 슬픔(위로),
    축하(승진), 축하(개업), 축하(합격), 축하(생일), 축하(출산),
    응원(새로운 시작), 응원(합격 기원), 응원(격려), 응원(꿈을 향한 도전),
    행복(영원한), 행복(순수한), 행복(함께한), 행복(다가올),
    특별함(비밀), 특별함(신비), 특별함(마법), 특별한(고귀), 특별한(고급)
    """
)
emotion_chain = LLMChain(llm=llm, prompt=emotion_prompt)

def classify_emotion(keywords: str) -> str:
    return emotion_chain.run({"keywords": keywords}).strip()


def expand_keywords(keywords: list[str], structured: bool = True) -> str:
    if structured and isinstance(keywords, list) and len(keywords) >= 6:
        target = keywords[0]
        gender = keywords[1]
        emotion_main = keywords[2]
        emotion_detail = keywords[3]
        personality = keywords[4]

        return (
            f"나는 성별이 {gender}인 {target}에게 {emotion_main}의 감정을 전하고 싶어요. "
            f"그 사람은 {personality}, 그래서 더욱 조심스럽고 진심을 담아 표현하고 싶어요. "
            f"{emotion_detail} {emotion_main}은 단순한 감정이 아니라, 그 사람과 나 사이에 오랜 시간 쌓여온 마음이에요. "
            f"말로 다 전할 수 없기에 꽃으로 대신 전하고 싶고, "
            f"이 꽃이 우리의 관계를 따뜻하게 이어주는 매개체가 되었으면 해요. "
            f"{target}에게 진심을 담아 건네는 이 꽃은 나의 마음 그 자체입니다."
        )

    # fallback 제거 → 입력이 부족하면 명확히 알림
    raise ValueError("키워드는 최소 4개의 요소(관계, 감정, 세부감정, 성향)를 포함해야 합니다.")


def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    emotion_category = classify_emotion(keywords)
    query_vector = embed_query(expanded_query)

    distances, indices = index.search(np.array(query_vector).astype("float32"), top_k * SEARCH_EXPANSION_FACTOR)

    results_with_score = []
    for i in indices[0]:
        flower = metadata_list[i]
        base_score = distances[0][list(indices[0]).index(i)]
        boost = -0.3 if emotion_category in flower.get("emotion_tags", []) else 0.0
        final_score = base_score + boost
        results_with_score.append((i, final_score))

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
        except Exception as e:
            reason = "[추천 이유 생성 실패]"

        final_results.append({
            "FLW_IDX": flower["FLW_IDX"],
            "reason": reason
        })

        if len(final_results) >= top_k:
            break

    return {"recommendations": final_results}
