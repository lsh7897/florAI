import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

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
    target, emotion, detail, personality, gender = keywords

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
        f"{personality} 성향의 사람에게 {emotion}({detail})이 어떻게 잘 표현할 수 있을 꽃을 찾고 있어요요."
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

        이 꽃이 '{query}'에 어울리는 이유를 구체적으로 설명해줘.
        꽃 이름({flower})도 반드시 포함하고, 꽃말을 중심으로 이꽃이 구매자가 당사자에게 어떠한 메세지를 보낼 수 있을지 설득력 있게 표현해줘.
        말투는 {emotion}에 맞춰서 조절해줘. 
        슬픔이면 조용하고 따뜻하게, 응원이면 희망차고 긍정적으로, 사랑이면 깊고 섬세하게, 축하면 경쾌하고 발랄하게, 특별함은 속삭이듯 비밀스럽게.
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

    desc_vec = embedder.embed_query(desc_query)
    emo_vec = embedder.embed_query(emo_query)
    meaning_vec = embedder.embed_query(style_query)

    SEARCH_TOP_K = 50  # 여유 있게 확보
    vectors = {"desc": desc_vec, "emotion": emo_vec, "meaning": meaning_vec}
    results = {
        name: qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": name, "vector": vector},
            limit=SEARCH_TOP_K
        )
        for name, vector in vectors.items()
    }

    weights = {"desc": 0.6, "emotion": 0.3, "meaning": 0.1}
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
    top_names = [x[0] for x in flower_scores]

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
