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

# 🔹 메타데이터 로드 (벡터 없음)
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

        이 키워드에서 느껴지는 중심 감정을 다음 중 하나로 분류해줘 (목록 외 감정은 절대 사용하지 마):

        사랑(강렬한), 사랑(순수한), 사랑(영원한), 사랑(행복한), 사랑(따뜻한),
        슬픔(후회), 슬픔(이별), 슬픔(그리움), 슬픔(위로),
        축하(승진), 축하(개업), 축하(졸업), 축하(결혼), 축하(출산),
        응원(새로운 시작), 응원(합격 기원), 응원(격려), 응원(꿈을 향한 도전),
        행복(영원한), 행복(순수한), 행복(함께한), 행복(다가올),
        특별함(비밀), 특별함(신비), 특별함(마법), 특별한(고귀), 특별한(고급)

        정확하게 위 목록 중 하나만 출력해. 이유는 쓰지 말고.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"keywords": keywords}).strip()

    VALID_CATEGORIES = {
        "사랑(강렬한)", "사랑(순수한)", "사랑(영원한)", "사랑(행복한)", "사랑(따뜻한)",
        "슬픔(후회)", "슬픔(이별)", "슬픔(그리움)", "슬픔(위로)",
        "축하(승진)", "축하(개업)", "축하(졸업)", "축하(결혼)", "축하(출산)",
        "응원(새로운 시작)", "응원(합격 기원)", "응원(격려)", "응원(꿈을 향한 도전)",
        "행복(영원한)", "행복(순수한)", "행복(함께한)", "행복(다가올)",
        "특별함(비밀)", "특별함(신비)", "특별함(마법)", "특별한(고귀)", "특별한(고급)"
    }
    return result if result in VALID_CATEGORIES else "이별(슬픔)"

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

    # 🔍 감정에 해당하는 꽃만 필터링 (index 기반)
    filtered_indices = [i for i, flower in enumerate(metadata_list)
                        if "emotion_tags" in flower and emotion_category in flower["emotion_tags"]]

    if not filtered_indices:
        return {
            "expanded_query": expanded_query,
            "emotion_category": emotion_category,
            "recommendations": [],
            "error": f"'{emotion_category}' 감정에 해당하는 꽃이 없습니다."
        }

    # 🔧 FAISS 임시 인덱스 구성 (필터된 것만)
    dim = index.d
    sub_index = faiss.IndexFlatL2(dim)
    sub_vectors = [index.reconstruct(i) for i in filtered_indices]
    sub_index.add(np.array(sub_vectors).astype("float32"))

    distances, sub_idxs = sub_index.search(np.array(query_vector).astype("float32"), len(filtered_indices))

    # 🔄 같은 이름의 꽃이 있으면 유사도 높은 순서대로 하나만 선택
    ranked_by_name = {}
    for sub_i in sub_idxs[0]:
        real_index = filtered_indices[sub_i]
        flower = metadata_list[real_index]
        name = flower["name"]
        if name not in ranked_by_name:
            ranked_by_name[name] = real_index

    final_indices = list(ranked_by_name.values())[:top_k]

    results = []
    for real_index in final_indices:
        flower = metadata_list[real_index]
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