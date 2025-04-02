import faiss
import json
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings

# 🔹 벡터 인데스 및 메타데이터 보내오기
index = faiss.read_index("flower_index.faiss")
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# 🔹 LangChain LLM 설정
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

def embed_query(query: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("환경변수 OPENAI_API_KEY가 비어 있어요!")

    # 👉 명시적으로 환경변수에서 읽어서 넘기기 (LangChain 버그 방지)
    embedder = OpenAIEmbeddings(openai_api_key=api_key)
    return embedder.embed_query(query).reshape(1, -1)

def generate_reason(query: str, description: str, flower_name: str):
    prompt = PromptTemplate(
        input_variables=["query", "description", "flower"],
        template="""
        사용자 의도: {query}
        꽃 설명: {description}
        이 꽃이 '{query}'에 어울리는 이유를 한 문장으로 설명해줘. 꽃 이름({flower})도 포함해서 자전시게 쓰줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "query": query,
        "description": description,
        "flower": flower_name
    }).strip()

def expand_keywords(keywords: str) -> str:
    prompt = PromptTemplate(
        input_variables=["keywords"],
        template="""
        사용자가 입력한 키워드: {keywords}
        이 키워드를 바탕으로 감정과 상황을 포함한 자전시된 문장으로 확장해줘.
        너무 긴지 않고, 의도가 잘 들어날 수 있게 말해줘.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(keywords).strip()

def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)
    query_vector = embed_query(expanded_query)
    distances, indices = index.search(query_vector, 10)

    results = []
    seen = set()
    for idx in indices[0]:
        flower = metadata_list[idx]
        if flower["name"] in seen:
            continue
        seen.add(flower["name"])

        reason = generate_reason(expanded_query, flower["description"], flower["name"])
        results.append({
            "name": flower["name"],
            "description": flower["description"],
            "color": flower["color"],
            "season": flower["season"],
            "scent": flower["scent"],
            "reason": reason
        })

        if len(results) == top_k:
            break

    return {
        "expanded_query": expanded_query,
        "recommendations": results
    }
