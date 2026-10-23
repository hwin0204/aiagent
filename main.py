import os
import io
import json
import asyncio
import traceback
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from pypdf import PdfReader
from supabase import create_client, Client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

app = FastAPI(title="도시 정책 전문가 RAG AI 서버")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_required_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise ValueError(f"필수 환경 변수 '{key}'가 설정되지 않았습니다. Cloud Run 환경 변수를 확인하세요.")
    return val

class StreamHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            await self.queue.put({"type": "token", "content": token})

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        await self.queue.put({"type": "done"})

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        await self.queue.put({"type": "error", "message": f"LLM 스트리밍 오류: {str(error)}"})

class SupabaseVectorRetriever(BaseRetriever):
    supabase_client: Any
    embeddings_model: Any
    top_k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        try:
            query_vector = self.embeddings_model.embed_query(query)
            rpc_response = self.supabase_client.rpc(
                "match_documents",
                {
                    "query_embedding": query_vector,
                    "match_threshold": 0.2,
                    "match_count": self.top_k,
                },
            ).execute()

            docs: List[Document] = []
            for row in rpc_response.data or []:
                metadata = row.get("metadata") or {}
                metadata["similarity"] = round(float(row.get("similarity", 0.0)), 4)
                docs.append(Document(page_content=row.get("content", ""), metadata=metadata))
            return docs
        except Exception as e:
            raise RuntimeError(f"Supabase 벡터 DB 검색 중 오류 발생: {str(e)}")

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Dict[str, str]] = []

@app.get("/health")
async def health_check():
    missing = [k for k in ["GOOGLE_API_KEY", "OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"] if not os.environ.get(k)]
    return {
        "status": "ok" if not missing else "warning",
        "message": "서버 정상 작동 중" if not missing else f"누락된 환경 변수: {', '.join(missing)}",
        "llm_model": "gemini-3.5-flash-lite"
    }

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    async def process_stream():
        try:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                yield json.dumps({"stage": "error", "progress": 0, "message": "유효한 PDF 파일이 아닙니다."}, ensure_ascii=False) + "\n"
                return

            yield json.dumps({"stage": "reading", "progress": 10, "message": f"'{file.filename}' 텍스트 추출 중..."}, ensure_ascii=False) + "\n"
            pdf_bytes = await file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))

            pages_data = []
            full_text_list = []
            for idx, page in enumerate(reader.pages):
                page_text = (page.extract_text() or "").strip()
                if page_text:
                    pages_data.append({"page": idx + 1, "text": page_text})
                    full_text_list.append(f"[페이지 {idx + 1}]\n{page_text}")

            full_text = "\n\n".join(full_text_list)
            if not full_text.strip():
                yield json.dumps({"stage": "error", "progress": 0, "message": "PDF에서 텍스트를 추출할 수 없습니다."}, ensure_ascii=False) + "\n"
                return

            yield json.dumps({
                "stage": "extracted",
                "progress": 30,
                "message": f"총 {len(pages_data)}페이지 추출 완료. 청크 분할 중...",
                "extracted_text": full_text
            }, ensure_ascii=False) + "\n"

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks_with_meta = []
            chunk_counter = 1
            for p in pages_data:
                for ch in splitter.split_text(p["text"]):
                    chunks_with_meta.append({
                        "content": ch,
                        "metadata": {"source": file.filename, "page": p["page"], "chunk_index": chunk_counter}
                    })
                    chunk_counter += 1

            yield json.dumps({"stage": "embedding", "progress": 55, "message": f"총 {len(chunks_with_meta)}개 청크 임베딩 생성 중..."}, ensure_ascii=False) + "\n"

            openai_key = get_required_env("OPENAI_API_KEY")
            supabase_url = get_required_env("SUPABASE_URL").replace("/rest/v1/", "").rstrip("/")
            supabase_key = get_required_env("SUPABASE_KEY")

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key)
            vectors = embeddings.embed_documents([c["content"] for c in chunks_with_meta])

            yield json.dumps({"stage": "storing", "progress": 80, "message": "Supabase 벡터 DB에 저장 중..."}, ensure_ascii=False) + "\n"

            supabase: Client = create_client(supabase_url, supabase_key)
            rows = [{"content": item["content"], "metadata": item["metadata"], "embedding": vec} for item, vec in zip(chunks_with_meta, vectors)]
            supabase.table("documents").insert(rows).execute()

            yield json.dumps({"stage": "completed", "progress": 100, "message": f"벡터 DB 생성 완료! ({len(rows)}개 청크 저장됨)"}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"stage": "error", "progress": 0, "message": f"오류: {str(e)}\n{traceback.format_exc()}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")

@app.post("/query_stream")
async def query_stream(req: QueryRequest):
    async def event_generator():
        try:
            if not req.query or not req.query.strip():
                yield json.dumps({"type": "error", "message": "질의 내용이 비어 있습니다."}, ensure_ascii=False) + "\n"
                return

            google_api_key = get_required_env("GOOGLE_API_KEY")
            openai_api_key = get_required_env("OPENAI_API_KEY")
            supabase_url = get_required_env("SUPABASE_URL").replace("/rest/v1/", "").rstrip("/")
            supabase_key = get_required_env("SUPABASE_KEY")

            chat_objects = []
            for turn in req.chat_history:
                if turn.get("role") == "user":
                    chat_objects.append(HumanMessage(content=turn.get("content", "")))
                elif turn.get("role") == "ai":
                    chat_objects.append(AIMessage(content=turn.get("content", "")))

            augmentation_system_msg = (
                "기존의 대화 내용을 활용하여 사용자가 질문한 의도를 파악해서 한문장의 명료한 질문으로 변환하라. "
                "대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라"
            )
            augmentation_prompt = ChatPromptTemplate.from_messages([
                ("system", augmentation_system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{query}")
            ])

            llm_for_aug = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=google_api_key, temperature=0.1)
            query_augmentation_chain = augmentation_prompt | llm_for_aug | StrOutputParser()

            augmented_query = (await query_augmentation_chain.ainvoke({
                "query": req.query,
                "chat_history": chat_objects
            })).strip()

            yield json.dumps({"type": "augmented_query", "original_query": req.query, "augmented_query": augmented_query}, ensure_ascii=False) + "\n"

            supabase: Client = create_client(supabase_url, supabase_key)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
            retriever = SupabaseVectorRetriever(supabase_client=supabase, embeddings_model=embeddings, top_k=4)

            retrieved_docs = await asyncio.to_thread(retriever.invoke, augmented_query)

            retrieval_list = []
            sources = []
            context_blocks = []
            for idx, doc in enumerate(retrieved_docs):
                meta = doc.metadata or {}
                src = f"{meta.get('source', '문서')} (p.{meta.get('page', '?')}, 청크 #{meta.get('chunk_index', idx+1)})"
                sources.append(src)
                retrieval_list.append({
                    "rank": idx + 1,
                    "content": doc.page_content,
                    "source": meta.get("source", "문서"),
                    "page": meta.get("page", "?"),
                    "chunk_index": meta.get("chunk_index", idx + 1),
                    "similarity": meta.get("similarity", 0.0)
                })
                context_blocks.append(f"[출처 {idx+1}: {src}]\n{doc.page_content}")

            yield json.dumps({"type": "retrieval_result", "retrieved_chunks": retrieval_list, "sources": list(dict.fromkeys(sources))}, ensure_ascii=False) + "\n"

            if not retrieved_docs:
                yield json.dumps({"type": "token", "content": "저장된 문서에서 관련 내용을 찾을 수 없습니다. 먼저 PDF 벡터 DB를 생성해 주세요."}, ensure_ascii=False) + "\n"
                yield json.dumps({"type": "final", "sources": []}, ensure_ascii=False) + "\n"
                return

            queue: asyncio.Queue = asyncio.Queue()
            stream_handler = StreamHandler(queue)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "너는 문서에 기반해 답변하는 도시 정책 전문가야.\n제공된 [문서 컨텍스트]에 기반하여 정확하게 답변하고 출처(파일명, 페이지)를 함께 언급해.\n\n[문서 컨텍스트]\n{context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "원래 질문: {original_query}\n확장된 질문: {augmented_query}\n\n위 질문에 대해 도시 정책 전문가로서 답변해줘.")
            ])

            streaming_llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=google_api_key,
                streaming=True,
                callbacks=[stream_handler],
                temperature=0.2
            )

            qa_chain = qa_prompt | streaming_llm

            task = asyncio.create_task(qa_chain.ainvoke({
                "context": "\n\n".join(context_blocks),
                "chat_history": chat_objects,
                "original_query": req.query,
                "augmented_query": augmented_query
            }))

            while True:
                item = await queue.get()
                if item["type"] == "token":
                    yield json.dumps(item, ensure_ascii=False) + "\n"
                elif item["type"] == "error":
                    yield json.dumps(item, ensure_ascii=False) + "\n"
                    break
                elif item["type"] == "done":
                    break

            await task
            yield json.dumps({"type": "final", "sources": list(dict.fromkeys(sources))}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"질의 처리 오류: {str(e)}\n{traceback.format_exc()}"}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
