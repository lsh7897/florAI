import os
import json
import faiss
import numpy as np
from app.utils import embed_query, generate_reason
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


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
    if structured and len(keywords) >= 5:
        target = keywords[0]
        gender = keywords[1]
        emotion_main = keywords[2]
        emotion_detail = keywords[3]
        personality = keywords[4]

        # 좀 더 풍부한 기본 문장 생성
        base_sentence = (
            f"나는 성별이 {gender}인 {target}에게 {emotion_main}의 감정을 전하고 싶어요. "
            f"그 사람은 {personality}한 성격을 가진 사람이고, {emotion_detail} {emotion_main}을 전하기에 적합한 사람이에요. "
            f"내가 {target}에게 전하고 싶은 감정은 단순한 말로는 다 표현할 수 없고, "
            f"그 사람에게 내 진심을 잘 전달할 수 있는 특별한 방법이 필요해요. "
            f"그래서 이 꽃을 통해 내 마음을 전하고 싶어요. 이 꽃이 우리의 관계에 큰 의미를 더해줄 것 같아요."
        )

        # **수정된 부분**: LLMChain을 활용한 문장 확장 프롬프트 정의
        expand_prompt = PromptTemplate(
            input_variables=["base_sentence"],
            template="""다음 문장을 감정의 깊이를 담아 4~6문장으로 확장해 주세요. 
                    문장은 감정이 잘 전달되고, 사실을 기반으로한 이야기기처럼 자연스럽고 부드럽게 이어져야 합니다. 
                    각 문장이 감정을 잘 표현할 수 있도록 구체적으로 만들어 주세요. 
                    또한, 감정을 잘 전달하면서도 그 사람의 성격이나 관계를 염두에 두고, 
                    전하려는 감정이 더 강하게 느껴지도록 신경 써 주세요.

                    문장: {base_sentence}
                    """
        )
        # **수정된 부분**: LLMChain 정의 (expand_chain)
        expand_chain = LLMChain(llm=llm, prompt=expand_prompt)

        # **수정된 부분**: 확장된 문장 생성 (확실하게 run() 메서드 호출)
        expanded = expand_chain.run({"base_sentence": base_sentence}).strip()

        return expanded

    raise ValueError("키워드는 최소 5개의 요소(관계, 성별, 감정, 세부감정, 성향)를 포함해야 합니다.")


def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    # 문장 확장 (4~6문장으로)
    expanded_query = expand_keywords(keywords)
    
    # 사용자의 감정 카테고리 추출 (슬픔, 그리움 등)
    emotion_category = classify_emotion(keywords)

    # emotion_tags에서 괄호 안의 내용 제거하고 태그명만 비교
    emotion_category_cleaned = emotion_category.split('(')[0].strip()

    # 유사도 계산을 위한 쿼리 벡터 생성
    query_vector = embed_query(expanded_query)
    
    # 유사도 검색 (상위 5개 꽃 먼저 검색)
    distances, indices = index.search(np.array(query_vector).astype("float32"), top_k * SEARCH_EXPANSION_FACTOR)

    results = []
    seen_names = set()

    # 슬픔, 그리움에 해당하는 꽃만 유사도로 추천
    for i in indices[0]:
        flower = metadata_list[i]
        
        # emotion_tags에서 괄호 내용을 제외하고 태그만 비교
        flower_tags = [tag.split('(')[0].strip() for tag in flower.get("emotion_tags", [])]

        # 슬픔, 그리움에 맞는 꽃만 필터링
        if emotion_category_cleaned in flower_tags:
            if flower["name"] in seen_names:
                continue
            seen_names.add(flower["name"])

            # 추천 이유 생성
            reason = generate_reason(expanded_query, flower["description"], flower["name"])
            results.append({
                "FLW_IDX": flower["FLW_IDX"],
                "reason": reason
            })

            if len(results) >= top_k:
                break

    # 감정에 맞는 꽃이 부족하면 다른 감정 태그를 가진 꽃 중에서 유사도 기반으로 채우기
    if len(results) < top_k:
        for i in indices[0]:
            flower = metadata_list[i]
            if flower["name"] in seen_names:
                continue
            seen_names.add(flower["name"])

            reason = generate_reason(expanded_query, flower["description"], flower["name"])
            results.append({
                "FLW_IDX": flower["FLW_IDX"],
                "reason": reason
            })

            if len(results) >= top_k:
                break

    return {"recommendations": results}




