
import os
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# Qdrant 클라이언트
qdrant = QdrantClient(
    url=os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
COLLECTION_NAME = "flowers"

# 임베딩
embedder = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-ada-002"
)

# LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo"
)

# 키워드 5개: 대상, 감정, 세부감정, 성향, 성별
def expand_query_desc(keywords: list[str]) -> str:
    base = (keywords + [""] * 5)[:5]
    target, emotion, detail, personality, gender = base
    return f"{target}에게 {emotion}({detail})의 감정을 표현하고 싶어요. 상대는 {gender}이고 {personality}입니다."

def generate_reason(query: str, description: str, flower_name: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower"],
        template="""
        사용자 의도: {query}
        꽃 설명: {description}
        이 꽃이 '{query}'에 어울리는 이유를 두 문장이상으로 설명해줘. 꽃 이름({flower})도 포함해서 구매자를 충분히 설득 할 수 있도록 구체적으로 설명해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name
    }).strip()

def get_flower_recommendations(keywords: list[str], top_k: int = 3):
    desc_query = expand_query_desc(keywords)
    desc_vec = embedder.embed_query(desc_query)

    print("📌 쿼리 문장:", desc_query)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "desc", "vector": desc_vec},
        limit=top_k * 5
    )

    print("\n📊 유사도 상위 결과:")
    for r in results[:5]:
        print(f"  - {r.payload['name']}: {r.score:.4f}")

    final = []
    seen = set()
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        payload = r.payload
        name = payload["name"]
        if name in seen:
            continue
        seen.add(name)

        reason = generate_reason(desc_query, payload["description"], name)
        final.append({
            "FLW_IDX": payload["FLW_IDX"],
            "name": name,
            "score": round(r.score, 4),
            "reason": reason
        })

        if len(final) >= top_k:
            break

    return {"recommendations": final}
