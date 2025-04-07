import os
import json
import faiss
import numpy as np
from app.utils import embed_query, generate_reason
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load FAISS index
index = faiss.read_index("flower_index.faiss")

# Load metadata
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# LLM for emotion classification
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

def classify_emotion(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
다음은 사용자가 꽃을 추천받기 위한 키워드입니다. 아래는 예시입니다:

예시 1:
입력: 연인, 사랑, 영원한, 감성적인 사람입니다  
→ 사랑

예시 2:
입력: 부모, 슬픔, 그리움, 차분한 사람입니다  
→ 슬픔

예시 3:
입력: 친구, 응원, 격려, 활발한 사람입니다  
→ 응원

예시 4:
입력: 동료, 축하, 승진, 성실한 사람입니다  
→ 축하

예시 5:
입력: 연인, 특별함, 마법같은, 감정적인 사람입니다  
→ 특별함

이제 사용자의 실제 키워드를 분석해서 아래 감정 카테고리 중 가장 적절한 **하나만** 골라줘.  
정확히 하나만 반환해줘 (괄호 포함해서).

입력: {keywords}

가능한 감정 카테고리:
- 사랑(강렬한), 사랑(순수한), 사랑(영원한), 사랑(행복한), 사랑(따뜻한)
- 슬픔(화해), 슬픔(이별), 슬픔(그리움), 슬픔(위로)
- 축하(승진), 축하(개업), 축하(합격), 축하(생일), 축하(출산)
- 응원(새로운 시작), 응원(합격 기원), 응원(격려), 응원(꿈을 향한 도전)
- 행복(영원한), 행복(순수한), 행복(함께한), 행복(다가올)
- 특별함(비밀), 특별함(신비), 특별함(마법), 특별함(고귀), 특별함(고급)
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"keywords": keywords}).strip()



def expand_keywords(keywords: list[str], structured: bool = True) -> str:
    if structured and isinstance(keywords, list) and len(keywords) >= 4:
        target = keywords[0]
        emotion_main = keywords[1]
        emotion_detail = keywords[2]
        personality = keywords[3]
        return (
            f"{emotion_detail} {emotion_main} 감정을 {target}에게 표현하고 싶어. "
            f"그 감정은 {personality}인 사람에게 깊은 인상을 줄 수 있을 거야."
        )

    # fallback for freeform input
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        사용자가 입력한 키워드: {keywords}
        이 키워드를 바탕으로 감정과 상황을 포함한 자연스럽고 모든 의도가 잘 전달되게 문장으로 확장해줘.
        너무 길지 않고, 말하고자 하는 목적이 잘 나타나게 해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({"keywords": ",".join(keywords)}).strip()

def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    emotion_category = classify_emotion(expanded_query)
    query_vector = embed_query(expanded_query)

    distances, indices = index.search(np.array(query_vector).astype("float32"), top_k * 10)

    seen_names = set()
    filtered_candidates = []
    fallback_candidates = []

    for i in indices[0]:
        flower = metadata_list[i]
        name = flower["name"]
        if name in seen_names:
            continue
        seen_names.add(name)

        base_score = distances[0][list(indices[0]).index(i)]

        # 1️⃣ 감정 태그 일치한 경우
        if emotion_category in flower.get("emotion_tags", []):
            score = base_score - 1.0  # 보너스 점수
            # 3️⃣ keyword가 description에 포함되어 있으면 추가 보너스
            if any(word in flower["description"] for word in keywords.split(",")):
                score -= 0.2
            filtered_candidates.append((i, score))
        else:
            # 2️⃣ fallback 후보 저장 (일치하지 않지만 후보로 유지)
            fallback_candidates.append((i, base_score + 0.5))

    # 감정 태그 일치한 후보 우선 정렬
    filtered_candidates.sort(key=lambda x: x[1])
    final_results = []

    for i, _ in filtered_candidates:
        flower = metadata_list[i]
        reason = generate_reason(expanded_query, flower["description"], flower["name"])
        final_results.append({
            "FLW_IDX": flower["FLW_IDX"],
            "reason": reason
        })
        if len(final_results) >= top_k:
            break

    # 만약 충분하지 않으면 fallback에서 채우기
    if len(final_results) < top_k:
        fallback_candidates.sort(key=lambda x: x[1])
        for i, _ in fallback_candidates:
            flower = metadata_list[i]
            if any(res["FLW_IDX"] == flower["FLW_IDX"] for res in final_results):
                continue
            reason = generate_reason(expanded_query, flower["description"], flower["name"])
            final_results.append({
                "FLW_IDX": flower["FLW_IDX"],
                "reason": reason
            })
            if len(final_results) >= top_k:
                break

    return {"recommendations": final_results}
