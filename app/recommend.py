import faiss
import json
import os
import numpy as np
from app.utils import embed_query, generate_reason
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔹 벡터 인덱스 및 메타데이터 불러오기
index = faiss.read_index("flower_index.faiss")
with open("flower_metadata.json", encoding="utf-8") as f:
    metadata_list = json.load(f)

# 🔹 LangChain LLM 설정
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# 🔹 키워드 확장 함수
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

def get_flower_recommendations(keywords: str, top_k: int = 3):
    expanded_query = expand_keywords(keywords)

    # ✅ 임베딩을 2차원 numpy 배열로 보장
    raw_vector = embed_query(expanded_query)
    query_vector = np.array(raw_vector).reshape(1, -1)

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