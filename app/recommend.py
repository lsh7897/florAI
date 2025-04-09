import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import random

load_dotenv()

# Qdrant 연결
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "flowers"

# 임베딩 & GPT
embedder = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002")
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# 감정 프롬프트 구성
def expand_query_components(keywords: list[str]):
    if len(keywords) < 5:
        keywords += [""] * (5 - len(keywords))
    gender, target, emotion, detail, personality= keywords

    desc = (
        f"{target}에게 {emotion}({detail})의 감정을 진심으로 전하고 싶어요. "
        f"그 사람은 {gender}이며, {personality} 성향을 가지고 있어요. "
        f"이 감정은 단순한 표현이 아니라, 마음 깊은 곳에서 우러나온 그사람에게 전하고 싶은 진심이에요."
    )

    emo = (
        f"표현하려는 핵심 감정은 '{emotion}'이고, 세부 감정은 '{detail}'입니다. "
        f"이 감정을 가장 정확하고 확실히히 전달할 수 있는 꽃을 찾고 있어요."
    )

    style = (
        f"{gender}이고 {personality} 성향의 사람이 {emotion}({detail})을  어떻게 느낄지"
        f"{emotion}({detail})이 어떻게 잘 표현할 수 있을 꽃을 찾고 있어요요."
    )

    return desc, emo, style


# GPT 설명 생성
def generate_reason(query: str, description: str, flower_name: str, flower_meaning: str, emotion: str, target: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower", "meaning", "emotion", "target"],
        template="""
        당신은 꽃 추천 전문가입니다. 아래 정보를 바탕으로, 구매자가 '{target}'에게 꽃을 선물하려는 상황에 맞게 추천 이유를 작성해주세요.

        [입력 정보]
        - 사용자 의도: {query}
        - 꽃 이름: {flower}
        - 꽃 설명: {description}
        - 꽃말(핵심 의미): {meaning}
        - 추천 어조 감정: {emotion} (예: 사랑, 응원, 슬픔, 축하, 행복, 특별함)

        [작성 지침]
        1. {flower}는 첫 문단에서 자연스럽게 등장시켜 주세요. 그 꽃의 상징성과 분위기를 간결하게 요약해 주세요.
        2. {meaning}은 중심 메시지로 삼되 감정이 담기게 풀어주세요. 단 그대로 반복하지 않고 창의적이고 자연스럽게 녹여주세요.
        3. 추천 이유는 구매자가 '{target}'에게 전하고 싶은 감정이 진심처럼 느껴지도록 써주세요.
        4. 전체 문장은 2~3개의 짧은 문단으로 나누고, 존댓말을 사용하세요.
        5. 감정({emotion})에 따라 다음 어조 스타일을 반영해 주세요:
        - 슬픔: 조용하고 따뜻하게
        - 응원: 희망차고 긍정적으로
        - 사랑: 깊고 섬세하게
        - 축하: 경쾌하고 발랄하게
        - 행복: 기쁘고 즐겁게
        - 특별함: 속삭이듯 비밀스럽게
    """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name,
        "meaning": flower_meaning,
        "emotion": emotion,
        "target": target
    }).strip()

# 추천 메인 함수
def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query, emo_query, style_query = expand_query_components(keywords)
    emotion = keywords[2] if len(keywords) >= 3 else ""
    target = keywords[1] if len(keywords) >= 2 else ""

    # 임베딩
    desc_vec = embedder.embed_query(desc_query)
    emo_vec = embedder.embed_query(emo_query)
    meaning_vec = embedder.embed_query(style_query)

    SEARCH_TOP_K = 50
    vectors = {"desc": desc_vec, "emotion": emo_vec, "meaning": meaning_vec}
    results = {
        name: qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": name, "vector": vector},
            limit=SEARCH_TOP_K
        )
        for name, vector in vectors.items()
    }

    # 가중 평균
    weights = {"desc": 0.4, "emotion": 0.4, "meaning": 0.2}
    score_map = {}
    for vector_name, result in results.items():
        for res in result:
            name = res.payload["name"]
            score = res.score
            score_map.setdefault(name, []).append((vector_name, score))

    flower_scores = []
    for name, scores in score_map.items():
        score_total = 0.0
        used = {"desc": 0.0, "emotion": 0.0, "meaning": 0.0}
        for vector_name, score in scores:
            used[vector_name] = score
        for k, v in weights.items():
            score_total += used[k] * v
        flower_scores.append((name, score_total))

    flower_scores.sort(key=lambda x: x[1], reverse=True)
    candidates = flower_scores[:30]  # 정확도 확보용 후보군

    # 다양성 그룹화 + 랜덤 추출
    grouped = []
    used = set()
    for name, score in candidates:
        if name in used:
            continue
        group = [(n, s) for n, s in candidates if abs(s - score) <= 0.03 and n not in used]
        chosen = random.choice(group)
        grouped.append(chosen)
        for n, _ in group:
            used.add(n)
        if len(grouped) >= top_k:
            break

    top_names = [x[0] for x in grouped]

    # 결과 생성
    final_recommendations = []
    seen = set()
    for name in top_names:
        for vector_name, result in results.items():
            for res in result:
                payload = res.payload
                if payload["name"] == name and name not in seen:
                    seen.add(name)

                    description = payload.get("description", "")
                    if isinstance(description, list):
                        description = " ".join(description)

                    try:
                        reason = generate_reason(
                            query=",".join(keywords),
                            description=description,
                            flower_name=name,
                            flower_meaning=description,
                            emotion=emotion,
                            target=target
                        )
                    except Exception as e:
                        print(f"❗ {name} GPT 설명 생성 오류:", e)
                        reason = f"{name}는 감정을 담아 표현하기 좋은 꽃이에요."

                    final_recommendations.append({
                        "FLW_IDX": payload["FLW_IDX"],
                        "name": name,
                        "score": round(score_map[name][0][1], 4),
                        "reason": reason
                    })
                    break
        if len(final_recommendations) >= top_k:
            break

    return {"recommendations": final_recommendations}

