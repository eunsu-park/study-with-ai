# 12. 실전 챗봇 프로젝트

## 학습 목표

- 대화형 AI 시스템 설계
- RAG 기반 챗봇 구현
- 대화 관리와 메모리
- 프로덕션 배포 고려사항

---

## 1. 챗봇 아키텍처

### 기본 구조

```
┌─────────────────────────────────────────────────────────────┐
│                       Chatbot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  사용자 입력                                                  │
│      │                                                       │
│      ▼                                                       │
│  [의도 분류] ──▶ FAQ / RAG / 일반 대화 분기                   │
│      │                                                       │
│      ▼                                                       │
│  [컨텍스트 검색] ◀── 벡터 DB                                 │
│      │                                                       │
│      ▼                                                       │
│  [프롬프트 구성] ◀── 대화 히스토리                           │
│      │                                                       │
│      ▼                                                       │
│  [LLM 생성]                                                  │
│      │                                                       │
│      ▼                                                       │
│  응답 출력                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 기본 챗봇 구현

### 간단한 대화 챗봇

```python
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, system_prompt=None):
        self.client = OpenAI()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.history = []

    def chat(self, user_message):
        # 메시지 구성
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        # API 호출
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear_history(self):
        self.history = []

# 사용
bot = SimpleChatbot("You are a friendly customer support agent.")
print(bot.chat("Hi, I need help with my order."))
print(bot.chat("My order number is 12345."))
```

---

## 3. RAG 챗봇

### 문서 기반 Q&A

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChatbot:
    def __init__(self, documents, persist_dir="./rag_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.embeddings = OpenAIEmbeddings()

        # 문서 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # 벡터 스토어
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # 대화 히스토리
        self.history = []

        # RAG 체인 구성
        self._setup_chain()

    def _setup_chain(self):
        template = """You are a helpful assistant. Answer based on the context.
If you don't know the answer, say so.

Context:
{context}

Conversation History:
{history}

Question: {question}

Answer:"""

        self.prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def format_history(history):
            if not history:
                return "No previous conversation."
            return "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "history": lambda x: format_history(self.history),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def chat(self, question):
        response = self.chain.invoke(question)

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_sources(self, question):
        """검색된 소스 문서 반환"""
        docs = self.retriever.invoke(question)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
```

---

## 4. 고급 대화 관리

### 의도 분류

```python
class IntentClassifier:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def classify(self, message, intents):
        prompt = f"""Classify the user message into one of these intents: {intents}

Message: {message}

Intent (only output the intent name):"""

        response = self.llm.invoke(prompt)
        return response.content.strip()

# 사용
classifier = IntentClassifier()
intent = classifier.classify(
    "I want to return my purchase",
    ["order_status", "return_request", "product_inquiry", "general"]
)
# "return_request"
```

### 슬롯 추출

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class OrderSlots(BaseModel):
    order_id: str = Field(default=None, description="Order ID")
    product_name: str = Field(default=None, description="Product name")
    issue: str = Field(default=None, description="Customer's issue")

class SlotExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=OrderSlots)

    def extract(self, message, context=""):
        prompt = f"""Extract information from the message.
{self.parser.get_format_instructions()}

Context: {context}
Message: {message}

JSON:"""

        response = self.llm.invoke(prompt)
        return self.parser.parse(response.content)

# 사용
extractor = SlotExtractor()
slots = extractor.extract("I want to return order #12345, the shirt is too small")
# {'order_id': '12345', 'product_name': 'shirt', 'issue': 'too small'}
```

### 대화 상태 관리

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any

class ConversationState(Enum):
    GREETING = "greeting"
    COLLECTING_INFO = "collecting_info"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    COMPLETED = "completed"

@dataclass
class ConversationContext:
    state: ConversationState = ConversationState.GREETING
    slots: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)

class StatefulChatbot:
    def __init__(self):
        self.context = ConversationContext()
        self.required_slots = ["order_id", "issue"]

    def process(self, message):
        # 슬롯 추출
        new_slots = self.extract_slots(message)
        self.context.slots.update({k: v for k, v in new_slots.items() if v})

        # 누락된 슬롯 확인
        self.context.missing_slots = [
            s for s in self.required_slots
            if s not in self.context.slots or not self.context.slots[s]
        ]

        # 상태 전이
        if self.context.missing_slots:
            self.context.state = ConversationState.COLLECTING_INFO
            return self.ask_for_slot(self.context.missing_slots[0])
        else:
            self.context.state = ConversationState.CONFIRMING
            return self.confirm_action()

    def ask_for_slot(self, slot_name):
        prompts = {
            "order_id": "Could you please provide your order number?",
            "issue": "What issue are you experiencing with your order?"
        }
        return prompts.get(slot_name, f"Please provide {slot_name}.")

    def confirm_action(self):
        return f"Let me confirm: Order {self.context.slots['order_id']}, Issue: {self.context.slots['issue']}. Is this correct?"
```

---

## 5. 스트리밍 응답

```python
from openai import OpenAI

class StreamingChatbot:
    def __init__(self):
        self.client = OpenAI()
        self.history = []

    def chat_stream(self, message):
        messages = [{"role": "system", "content": "You are helpful."}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        stream = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True
        )

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # 히스토리 업데이트
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

# 사용
bot = StreamingChatbot()
for chunk in bot.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 6. FastAPI 웹 서버

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# 세션 저장소
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 세션 가져오기/생성
    if request.session_id not in sessions:
        sessions[request.session_id] = RAGChatbot(documents)

    bot = sessions[request.session_id]

    # 응답 생성
    response = bot.chat(request.message)
    sources = bot.get_sources(request.message)

    return ChatResponse(response=response, sources=sources)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = StreamingChatbot()

    bot = sessions[request.session_id]

    def generate():
        for chunk in bot.chat_stream(request.message):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 7. Gradio UI

```python
import gradio as gr

class ChatbotUI:
    def __init__(self):
        self.bot = RAGChatbot(documents)

    def respond(self, message, history):
        response = self.bot.chat(message)
        return response

    def launch(self):
        demo = gr.ChatInterface(
            fn=self.respond,
            title="Document Q&A Chatbot",
            description="Ask questions about your documents",
            examples=["What is this document about?", "Summarize the main points"],
            theme="soft"
        )
        demo.launch()

# 사용
ui = ChatbotUI()
ui.launch()
```

---

## 8. 프로덕션 고려사항

### 에러 처리

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionChatbot:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def chat(self, message):
        try:
            response = self._generate_response(message)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
```

### 토큰/비용 관리

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def truncate_history(self, history, max_history_tokens=2000):
        """오래된 메시지부터 제거"""
        total_tokens = 0
        truncated = []

        for msg in reversed(history):
            msg_tokens = self.count_tokens(msg['content'])
            if total_tokens + msg_tokens > max_history_tokens:
                break
            truncated.insert(0, msg)
            total_tokens += msg_tokens

        return truncated
```

### 모니터링

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChatMetrics:
    session_id: str
    message: str
    response: str
    latency_ms: float
    token_count: int
    timestamp: float

class MonitoredChatbot:
    def __init__(self):
        self.metrics = []

    def chat(self, session_id, message):
        start = time.time()

        response = self._generate(message)

        latency = (time.time() - start) * 1000

        # 메트릭 기록
        metric = ChatMetrics(
            session_id=session_id,
            message=message,
            response=response,
            latency_ms=latency,
            token_count=self.token_manager.count_tokens(response),
            timestamp=time.time()
        )
        self.metrics.append(metric)

        return response

    def get_avg_latency(self):
        if not self.metrics:
            return 0
        return sum(m.latency_ms for m in self.metrics) / len(self.metrics)
```

---

## 9. 전체 시스템 예제

```python
"""
완전한 RAG 챗봇 시스템
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionRAGChatbot:
    def __init__(self, docs_dir, persist_dir="./prod_db"):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.embeddings = OpenAIEmbeddings()

        # 문서 로드
        logger.info(f"Loading documents from {docs_dir}")
        loader = DirectoryLoader(docs_dir, glob="**/*.txt")
        documents = loader.load()

        # 청킹
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # 벡터 스토어
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        # 세션 관리
        self.sessions = {}

    def _get_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"history": [], "context": {}}
        return self.sessions[session_id]

    def chat(self, session_id, message):
        session = self._get_session(session_id)

        # 관련 문서 검색
        docs = self.retriever.invoke(message)
        context = "\n\n".join([d.page_content for d in docs])

        # 히스토리 포맷
        history_text = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in session["history"][-6:]
        ])

        # 프롬프트
        prompt = f"""You are a helpful assistant. Answer based on the context provided.
If you cannot find the answer in the context, say so honestly.

Context:
{context}

Conversation History:
{history_text}

User: {message}
Assistant:"""

        # LLM 호출
        response = self.llm.invoke(prompt)
        answer = response.content

        # 히스토리 업데이트
        session["history"].append({"role": "user", "content": message})
        session["history"].append({"role": "assistant", "content": answer})

        return answer
```

---

## 정리

### 챗봇 설계 체크리스트

```
□ 용도 정의 (일반 대화 / FAQ / 문서 기반)
□ RAG 필요 여부 결정
□ 대화 히스토리 관리 방식
□ 의도 분류 필요 여부
□ 에러 처리 및 폴백
□ 비용 관리 (토큰 제한)
□ 모니터링 및 로깅
```

### 핵심 패턴

```python
# 기본 챗봇
messages = [system_prompt] + history + [user_message]
response = llm.invoke(messages)

# RAG 챗봇
docs = retriever.invoke(query)
context = format_docs(docs)
response = llm.invoke(prompt.format(context=context, question=query))

# 스트리밍
for chunk in llm.stream(messages):
    yield chunk
```

### 다음 단계

- 실제 서비스 배포 (AWS, GCP)
- A/B 테스트 설정
- 사용자 피드백 수집
- 지속적인 모델 개선

---

## 학습 완료

이것으로 LLM & NLP 학습 과정을 완료했습니다!

### 학습 요약

1. **NLP 기초 (01-03)**: 토큰화, 임베딩, Transformer
2. **사전학습 모델 (04-07)**: BERT, GPT, HuggingFace, 파인튜닝
3. **LLM 활용 (08-12)**: 프롬프트, RAG, LangChain, 벡터 DB, 챗봇

### 다음 단계 추천

- 실제 프로젝트에 적용
- Kaggle NLP 대회 참가
- 최신 LLM 논문 읽기 (Claude, Gemini, Llama)

