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
        raise ValueError("입력된 키워드는 최소 5개여야 합니다.")
    gender, target, emotion, detail, personality = keywords

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
        f"{gender}이고 {personality} 성향의 사람이 {emotion}({detail})을 어떻게 느낄지"
        f"{emotion}({detail})이 어떻게 잘 표현할 수 있을 꽃을 찾고 있어요."
    )

    return desc, emo, style


# GPT 설명 생성
def generate_reason(query: str, description: str, flower_name: str, flower_meaning: str, emotion: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower", "meaning", "emotion"],
        template="""
        사용자 의도: {query}
        꽃 설명: {description}
        꽃말: {meaning}

        아래 조건에 맞게 이 꽃이 '{query}'에 어울리는 이유를 설명해줘:

        1. 꽃 이름({flower})을 초반에 언급하고, 이 꽃이 가진 상징적인 의미와 분위기를 요약해줘.
        2. '꽃말'은 중심 메시지로 삼되, 반복하거나 뻔하게 말하지 말고 감정을 녹여서 자연스럽게 표현해줘.
        3. 구매자가 상대방에게 전하고 싶은 감정이 진심처럼 느껴지도록, 감정선을 잘 이어줘.
        4. 전체 문장은 너무 길지 않게, 핵심 중심으로 문단 2~3개로 나눠줘.

        말투는 '{emotion}' 감정에 맞춰서 다음 스타일을 따라줘:
        - 슬픔: 조용하고 따뜻하게
        - 응원: 희망차고 긍정적으로
        - 사랑: 깊고 섬세하게
        - 축하: 경쾌하고 발랄하게
        - 특별함: 속삭이듯 비밀스럽게
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name,
        "meaning": flower_meaning,
        "emotion": emotion
    }).strip()

# 추천 메인 함수
def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query, emo_query, style_query = expand_query_components(keywords)
    emotion = keywords[1] if len(keywords) >= 2 else ""

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
    flower_scores = []
    for vector_name, result in results.items():
        for res in result:
            name = res.payload["name"]
            score = res.score
            flower_scores.append((name, vector_name, score))

    # 가중 평균 계산
    final_scores = {}
    for name, vector_name, score in flower_scores:
        if name not in final_scores:
            final_scores[name] = {"desc": 0, "emotion": 0, "meaning": 0}
        final_scores[name][vector_name] += score

    weighted_scores = []
    for name, scores in final_scores.items():
        score_total = sum([scores[k] * weights[k] for k in scores])
        weighted_scores.append((name, score_total))

    weighted_scores.sort(key=lambda x: x[1], reverse=True)

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
                            emotion=emotion
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

