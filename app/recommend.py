import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any
from app.utils import embed_query, generate_reason
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 경로 설정
INDEX_PATH = "flower_index.faiss"
METADATA_PATH = "flower_metadata.json"
SEARCH_EXPANSION_FACTOR = 7  # 더 많은 후보를 검색
EMOTION_WEIGHT = 0.7  # 감정 일치 가중치
SIMILARITY_WEIGHT = 0.3  # 벡터 유사도 가중치

# 인덱스 및 메타데이터 로드
index = faiss.read_index(INDEX_PATH)

# 메타데이터 로드 및 전처리
with open(METADATA_PATH, encoding="utf-8") as f:
    metadata_list = json.load(f)

# 감정 태그 전처리를 위한 맵 생성
emotion_map = {}
for flower in metadata_list:
    for tag in flower.get("emotion_tags", []):
        clean_tag = tag.split('(')[0].strip()
        detail = tag.split('(')[1].rstrip(')') if '(' in tag else ""
        
        if clean_tag not in emotion_map:
            emotion_map[clean_tag] = []
        
        emotion_map[clean_tag].append({
            "detail": detail,
            "flower_idx": flower["FLW_IDX"]
        })

# LLM 초기화
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo-16k")

# 감정 분류 프롬프트 개선
emotion_prompt = PromptTemplate(
    input_variables=["keywords", "query_context"],
    template="""
    다음은 꽃을 추천받기 위한 키워드와 상황입니다:
    
    키워드: {keywords}
    상황 설명: {query_context}
    
    위 정보를 바탕으로 다음 감정 카테고리 중에서 가장 적절한 것을 골라주세요.
    여러 감정이 복합되어 있다면, 주된 감정 하나와 부차적인 감정 하나를 함께 알려주세요.
    
    감정 카테고리:
    - 사랑: 강렬한, 순수한, 영원한, 행복한, 따뜻한
    - 슬픔: 화해, 이별, 그리움, 위로
    - 축하: 승진, 개업, 합격, 생일, 출산
    - 응원: 새로운 시작, 합격 기원, 격려, 꿈을 향한 도전
    - 행복: 영원한, 순수한, 함께한, 다가올
    - 특별함: 비밀, 신비, 마법, 고귀, 고급
    
    형식: [주감정(세부)], [부차적 감정(세부)]
    예시: 사랑(따뜻한), 응원(격려)
    """
)
emotion_chain = LLMChain(llm=llm, prompt=emotion_prompt)

# 키워드 확장 프롬프트 개선
expand_prompt = PromptTemplate(
    input_variables=["keywords", "emotion"],
    template="""
    다음 키워드와 감정을 바탕으로, 꽃을 선물하는 상황을 자세하고 감정이 풍부하게 설명하는 문단을 작성해주세요.
    문단은 5-7개의 짧고 자연스러운 문장으로 구성되어야 합니다.
    
    키워드: {keywords}
    주요 감정: {emotion}
    
    다음과 같은 내용을 포함하되, 자연스럽게 통합해주세요:
    1. 선물하는 대상과의 관계나 상황
    2. 전달하고 싶은 감정의 깊이와 뉘앙스
    3. 꽃을 통해 표현하고 싶은 마음
    4. 선물 받는 사람의 성격이나 특성(알고 있는 경우)
    
    작성 시 주의사항:
    - 과장된 표현보다는 진솔하고 따뜻한 감정 표현을 사용해주세요
    - 구체적인 상황이나 기억을 언급하면 더 좋습니다
    - 일반적인 내용보다는 키워드에 특화된 내용으로 작성해주세요
    """
)
expand_chain = LLMChain(llm=llm, prompt=expand_prompt)

# 추천 이유 생성 프롬프트 개선
reason_prompt = PromptTemplate(
    input_variables=["situation", "flower_info", "flower_name"],
    template="""
    다음은 꽃을 선물하려는 상황과 추천하려는 꽃에 대한 정보입니다:
    
    상황: {situation}
    꽃 이름: {flower_name}
    꽃 정보: {flower_info}
    
    위 정보를 바탕으로, 이 꽃이 해당 상황에 왜 적합한지 설득력 있게 설명하는 추천 이유를 3-4문장으로 작성해주세요.
    꽃말, 색상, 의미 등을 연결지어 설명하면 좋습니다.
    """
)
reason_chain = LLMChain(llm=llm, prompt=reason_prompt)

def parse_structured_keywords(keywords: List[str]) -> Dict[str, str]:
    """키워드 목록에서 구조화된 정보 추출"""
    result = {}
    
    if len(keywords) >= 1:
        result["target"] = keywords[0]
    if len(keywords) >= 2:
        result["gender"] = keywords[1]
    if len(keywords) >= 3:
        result["emotion_main"] = keywords[2]
    if len(keywords) >= 4:
        result["emotion_detail"] = keywords[3]
    if len(keywords) >= 5:
        result["personality"] = keywords[4]
        
    return result

def expand_keywords(keywords: List[str]) -> str:
    """키워드를 확장하여 상세한 상황 설명으로 변환"""
    # 키워드 구조화
    keyword_info = parse_structured_keywords(keywords)
    
    # 감정 분류
    keywords_str = ", ".join(keywords)
    emotion_result = emotion_chain.run({
        "keywords": keywords_str,
        "query_context": "고객이 꽃을 추천받기 위해 입력한 키워드입니다."
    })
    
    # 감정 정보에서 주요 감정 추출
    main_emotion = emotion_result.split(',')[0].strip()
    
    # 확장된 상황 설명 생성
    expanded_text = expand_chain.run({
        "keywords": keywords_str,
        "emotion": main_emotion
    })
    
    return expanded_text, main_emotion

def custom_generate_reason(situation: str, flower_info: str, flower_name: str) -> str:
    """꽃 추천 이유 생성"""
    return reason_chain.run({
        "situation": situation,
        "flower_info": flower_info,
        "flower_name": flower_name
    })

def calculate_combined_score(emotion_match: bool, distance: float) -> float:
    """감정 일치와 벡터 유사도를 결합한 점수 계산"""
    # 거리를 유사도로 변환 (거리가 작을수록 유사도가 높음)
    similarity = 1.0 / (1.0 + distance)
    
    # 감정 일치 가중치 적용
    emotion_score = 1.0 if emotion_match else 0.0
    
    # 최종 점수 계산
    return (EMOTION_WEIGHT * emotion_score) + (SIMILARITY_WEIGHT * similarity)


def get_flower_recommendations(keywords: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """키워드 기반 꽃 추천 함수"""
    # 키워드가 충분한지 확인
    if len(keywords) < 3:
        raise ValueError("키워드는 최소 3개 이상이어야 합니다. (대상, 감정, 세부감정 등)")
    
    # 키워드 확장 및 감정 분류
    expanded_query, emotion_category = expand_keywords(keywords)
    
    # 감정 카테고리에서 주요 감정 추출 (괄호 부분 제거)
    main_emotion = emotion_category.split('(')[0].strip()
    
    # 쿼리 벡터 생성
    query_vector = embed_query(expanded_query)
    
    # 유사도 검색 (후보 꽃 검색)
    distances, indices = index.search(np.array([query_vector]).astype("float32"), top_k * SEARCH_EXPANSION_FACTOR)
    
    # 결과 처리를 위한 준비
    candidates = []
    seen_names = set()
    
    # 후보 꽃 처리
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata_list):
            continue
            
        flower = metadata_list[idx]
        if flower["name"] in seen_names:
            continue
            
        # 이미 처리한 꽃 추가
        seen_names.add(flower["name"])
        
        # 감정 태그 확인
        flower_emotions = [tag.split('(')[0].strip() for tag in flower.get("emotion_tags", [])]
        emotion_match = main_emotion in flower_emotions
        
        # 결합 점수 계산
        combined_score = calculate_combined_score(emotion_match, distances[0][i])
        
        # 후보에 추가
        candidates.append({
            "flower": flower,
            "distance": distances[0][i],
            "emotion_match": emotion_match,
            "combined_score": combined_score
        })
    
    # 결합 점수 기준으로 정렬
    candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 최종 추천 결과 생성
    results = []
    for candidate in candidates[:top_k]:
        flower = candidate["flower"]
        
        # 추천 이유 생성
        reason = custom_generate_reason(
            expanded_query, 
            flower["description"], 
            flower["name"]
        )
        
        results.append({
            "FLW_IDX": flower["FLW_IDX"],
            "name": flower["name"],  # 꽃 이름 추가
            "reason": reason,
            "score": candidate["combined_score"],
            "emotion_match": candidate["emotion_match"]
        })
    
    return {
        "recommendations": results,
        "query_context": expanded_query,
        "emotion_category": emotion_category
    }