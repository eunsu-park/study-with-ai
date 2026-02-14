# 10. LangChain 기초

> **버전 정보**: 이 레슨은 LangChain 0.2+ (2024년~) 기준으로 작성되었습니다.
>
> LangChain은 빠르게 발전하는 라이브러리입니다. 주요 변경사항:
> - **LCEL (LangChain Expression Language)**: 권장 체인 구성 방식
> - **langchain-core, langchain-community**: 패키지 분리
> - **ConversationChain 대신 RunnableWithMessageHistory 권장**
>
> 최신 문서: https://python.langchain.com/docs/

## 학습 목표

- LangChain 핵심 개념
- LLM 래퍼와 프롬프트
- 체인과 에이전트
- 메모리 시스템
- LCEL (LangChain Expression Language) 심화
- LangGraph 기초

---

## 1. LangChain 개요

### 설치

```bash
# LangChain 0.2+
pip install langchain langchain-openai langchain-community
```

### 핵심 구성요소

```
LangChain
├── Models          # LLM 래퍼
├── Prompts         # 프롬프트 템플릿
├── Chains          # 순차적 호출
├── Agents          # 도구 사용 에이전트
├── Memory          # 대화 기록
├── Retrievers      # 문서 검색
└── Callbacks       # 모니터링
```

---

## 2. LLM 래퍼

### ChatOpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# 간단한 호출
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### 다양한 LLM

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")

# HuggingFace
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.1")

# Ollama (로컬)
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### 메시지 타입

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
]

response = llm.invoke(messages)
print(response.content)
```

---

## 3. 프롬프트 템플릿

### 기본 템플릿

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}."
)

prompt = template.format(topic="spring")
response = llm.invoke(prompt)
```

### Chat 프롬프트

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
    ("human", "{text}")
])

messages = template.format_messages(
    input_language="English",
    output_language="Korean",
    text="Hello, how are you?"
)

response = llm.invoke(messages)
```

### Few-shot 프롬프트

```python
from langchain_core.prompts import FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"},
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of every input:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

prompt = few_shot_prompt.format(word="big")
```

---

## 4. 체인 (Chains)

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 체인 구성
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# 파이프 연산자로 연결
chain = prompt | llm | output_parser

# 실행
result = chain.invoke({"topic": "programmers"})
print(result)
```

### 순차 체인

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 첫 번째 체인: 주제 생성
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a random topic for a story."
)

# 두 번째 체인: 스토리 작성
story_prompt = ChatPromptTemplate.from_template(
    "Write a short story about: {topic}"
)

# 체인 연결
chain = (
    {"topic": topic_prompt | llm | StrOutputParser()}
    | story_prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({})
```

### 병렬 체인

```python
from langchain_core.runnables import RunnableParallel

# 병렬 실행
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Long article here..."})
# {'summary': '...', 'keywords': '...', 'sentiment': '...'}
```

---

## 5. 출력 파서

### String Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | llm | parser  # 문자열로 변환
```

### JSON Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")

parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person info. {format_instructions}"),
    ("human", "{text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 25 years old"})
# {'name': 'John', 'age': 25}
```

### 구조화된 출력

```python
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieReview)
```

---

## 6. 에이전트 (Agents)

### 기본 에이전트

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# 도구 정의
search = DuckDuckGoSearchRun()
tools = [search]

# ReAct 프롬프트 로드
prompt = hub.pull("hwchase17/react")

# 에이전트 생성
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "What is the weather in Seoul?"})
```

### 커스텀 도구

```python
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [calculate, get_current_time]
```

### Tool 클래스

```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import Field

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search for information on the internet"

    def _run(self, query: str) -> str:
        # 검색 로직
        return f"Search results for: {query}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
```

---

## 7. 메모리 (Memory)

> **권장 방식 변경**: LangChain 0.2+에서는 `ConversationChain`, `ConversationBufferMemory` 등이
> deprecated 되었습니다. 새 프로젝트에서는 **RunnableWithMessageHistory** (아래 참조)를 사용하세요.

### (Legacy) 대화 버퍼 메모리

> ⚠️ **Deprecated**: 아래 "LCEL에서 메모리" 섹션의 `RunnableWithMessageHistory` 사용 권장

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 대화
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")
# "Your name is John"
```

### (Legacy) 요약 메모리

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# 긴 대화를 요약하여 저장
```

### (Legacy) 윈도우 메모리

```python
from langchain.memory import ConversationBufferWindowMemory

# 최근 k개의 대화만 유지
memory = ConversationBufferWindowMemory(k=5)
```

### LCEL에서 메모리 (권장)

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 사용
response = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user123"}}
)
```

---

## 8. RAG with LangChain

### 문서 로더

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)

# 텍스트 파일
loader = TextLoader("document.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 웹페이지
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### 텍스트 분할

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
```

### 벡터 스토어

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 검색
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is machine learning?")
```

### RAG 체인

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

template = """Answer based on the context:
Context: {context}
Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is machine learning?")
```

---

## 9. 스트리밍

```python
# 스트리밍 출력
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# 비동기 스트리밍
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 10. LCEL (LangChain Expression Language) 심화

LCEL은 LangChain 0.2+에서 체인을 구축하는 권장 방식입니다. 복잡한 LLM 애플리케이션을 구축하기 위한 선언적이고 조합 가능한 문법을 제공합니다.

### 파이프 연산자를 통한 체인 구성

파이프 연산자(`|`)는 컴포넌트를 왼쪽에서 오른쪽으로 연결합니다:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 각 컴포넌트는 "Runnable"
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# 파이프 연산자로 구성
chain = prompt | llm | output_parser

# 실행
result = chain.invoke({"topic": "programmers"})
```

### 핵심 Runnable 컴포넌트

#### RunnablePassthrough

입력을 그대로 전달하며, 데이터 라우팅에 유용합니다:

```python
from langchain_core.runnables import RunnablePassthrough

# 전체 입력 전달
chain = RunnablePassthrough() | llm

# 특정 필드 전달
chain = {"text": RunnablePassthrough()} | prompt | llm
```

#### RunnableParallel

여러 체인을 병렬로 실행합니다:

```python
from langchain_core.runnables import RunnableParallel

summary_chain = summary_prompt | llm | StrOutputParser()
keyword_chain = keyword_prompt | llm | StrOutputParser()
sentiment_chain = sentiment_prompt | llm | StrOutputParser()

# 세 체인을 병렬로 실행
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Long article text here..."})
# {'summary': '...', 'keywords': [...], 'sentiment': 'positive'}
```

#### RunnableLambda

임의의 함수를 Runnable로 래핑합니다:

```python
from langchain_core.runnables import RunnableLambda

def extract_text(data):
    """입력에서 텍스트 필드 추출."""
    return data["text"].upper()

chain = RunnableLambda(extract_text) | llm
result = chain.invoke({"text": "hello world"})
```

### LCEL에서 스트리밍

LCEL은 여러 스트리밍 모드를 지원합니다:

```python
# 동기 스트리밍
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# 비동기 스트리밍
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# 이벤트 스트리밍 (상세 스트리밍)
async for event in chain.astream_events({"topic": "AI"}, version="v1"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="", flush=True)
```

### 비교: 구식 체인 스타일 vs LCEL

#### 구식 스타일 (Deprecated)

```python
from langchain.chains import LLMChain

# 구식 방식
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="AI")
```

#### LCEL 스타일 (권장)

```python
# LCEL 방식
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "AI"})
```

**LCEL의 장점:**
- **조합성**: 컴포넌트를 쉽게 결합하고 재사용
- **스트리밍**: 스트리밍 출력 기본 지원
- **비동기**: 1급 비동기 지원
- **병렬화**: 가능한 경우 자동 병렬 실행
- **타입 안정성**: 더 나은 IDE 지원 및 에러 메시지

### 예제: LCEL을 사용한 RAG 체인

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 설정
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 프롬프트 템플릿
template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL 스타일 RAG 체인
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 실행
answer = rag_chain.invoke("What is machine learning?")

# 답변 스트리밍
for chunk in rag_chain.stream("What is deep learning?"):
    print(chunk, end="", flush=True)
```

### 심화: 분기와 라우팅

```python
from langchain_core.runnables import RunnableBranch

# 입력에 따라 라우팅
branch = RunnableBranch(
    (lambda x: "code" in x["topic"], code_chain),
    (lambda x: "math" in x["topic"], math_chain),
    default_chain  # 기본값
)

chain = {"topic": RunnablePassthrough()} | branch | llm
```

---

## 11. LangGraph 기초

**LangGraph**는 LLM을 사용하여 상태 유지형 다중 에이전트 애플리케이션을 구축하기 위한 라이브러리입니다. 그래프 기반 워크플로우로 LangChain을 확장합니다.

### LangGraph란?

LangGraph를 사용하면 애플리케이션을 그래프로 정의할 수 있습니다:
- **노드**는 함수 (LLM 호출, 도구 사용, 커스텀 로직)
- **엣지**는 노드 간의 흐름 정의
- **상태**는 그래프 실행 전체에서 유지됨

**LangGraph를 사용해야 하는 경우 (vs 체인):**

| 체인(LCEL) 사용 | LangGraph 사용 |
|----------------|----------------|
| 선형 워크플로우 | 사이클, 루프 |
| 간단한 분기 | 복잡한 라우팅 |
| 상태 없음 | 상태 유지 에이전트 |
| 단일 에이전트 | 다중 에이전트 시스템 |

### 설치

```bash
pip install langgraph
```

### StateGraph 개념

LangGraph는 노드를 통과하면서 상태를 유지하는 `StateGraph`를 사용합니다:

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

# 상태 스키마 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str

# 그래프 생성
graph = StateGraph(AgentState)
```

### 노드와 엣지

```python
from langchain_core.messages import HumanMessage, AIMessage

def agent_node(state: AgentState):
    """에이전트 결정 노드."""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages + [response], "next": "tool"}

def tool_node(state: AgentState):
    """도구 실행 노드."""
    # 도구 실행
    result = "Tool result here"
    return {"messages": state["messages"] + [AIMessage(content=result)], "next": END}

# 노드 추가
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)

# 엣지 추가
graph.add_edge("agent", "tool")
graph.add_edge("tool", END)

# 진입점 설정
graph.set_entry_point("agent")

# 컴파일
app = graph.compile()

# 실행
result = app.invoke({"messages": [HumanMessage(content="Hello")]})
```

### 도구 사용이 있는 간단한 에이전트

```python
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage

# 도구 정의
@tool
def search(query: str) -> str:
    """정보 검색."""
    return f"Search results for: {query}"

tools = [search]
llm_with_tools = llm.bind_tools(tools)

# 상태
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages"]

# 에이전트 노드
def call_agent(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# 도구 노드
def call_tool(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # 도구 실행
    tool_calls = last_message.tool_calls
    results = []
    for tool_call in tool_calls:
        tool_result = search.invoke(tool_call["args"])
        results.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    return {"messages": messages + results}

# 그래프 구축
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_node("tools", call_tool)

# 조건부 라우팅
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

# 컴파일 및 실행
app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Search for LangChain news")]})

# 대화 출력
for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

### 조건부 라우팅

LangGraph는 동적 라우팅을 위한 조건부 엣지를 지원합니다:

```python
def route_decision(state: AgentState):
    """상태에 따라 다음 노드 결정."""
    if state.get("error"):
        return "error_handler"
    elif state.get("needs_review"):
        return "review"
    else:
        return "complete"

graph.add_conditional_edges(
    "process",
    route_decision,
    {
        "error_handler": "error_handler",
        "review": "review",
        "complete": END
    }
)
```

### 시각화

LangGraph는 그래프를 시각화할 수 있습니다:

```python
from IPython.display import Image, display

# 그래프 구조 시각화
display(Image(app.get_graph().draw_mermaid_png()))
```

### 다중 에이전트 예제

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    messages: Sequence[BaseMessage]
    current_agent: str

def researcher(state: MultiAgentState):
    # 연구 에이전트
    return {"messages": [...], "current_agent": "writer"}

def writer(state: MultiAgentState):
    # 작성 에이전트
    return {"messages": [...], "current_agent": "reviewer"}

def reviewer(state: MultiAgentState):
    # 검토 에이전트
    return {"messages": [...], "current_agent": END}

# 다중 에이전트 그래프 구축
graph = StateGraph(MultiAgentState)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)

graph.add_edge("researcher", "writer")
graph.add_edge("writer", "reviewer")
graph.add_edge("reviewer", END)
graph.set_entry_point("researcher")

app = graph.compile()
```

### 주요 LangGraph 개념

- **체크포인팅**: 언제든지 상태 저장/복원
- **Human-in-the-loop**: 계속하기 전 사람의 승인을 위해 일시 중지
- **타임 트래블**: 모든 체크포인트에서 재생
- **영속성**: 대화 상태를 데이터베이스에 저장

---

## 정리

### 핵심 패턴

```python
# 기본 LCEL 체인
chain = prompt | llm | output_parser

# LCEL을 사용한 RAG 체인
rag_chain = (
    RunnableParallel(context=retriever, question=RunnablePassthrough())
    | prompt | llm | parser
)

# 에이전트 (전통적 방식)
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# 에이전트 (LangGraph)
graph = StateGraph(AgentState)
graph.add_node("agent", call_agent)
graph.add_conditional_edges("agent", should_continue)
app = graph.compile()
```

### 컴포넌트 선택 가이드

| 상황 | 컴포넌트 |
|------|----------|
| 단순 호출 | LLM + Prompt |
| 순차 처리 | Chain (LCEL) |
| 병렬 실행 | RunnableParallel |
| 문서 기반 Q&A | RAG Chain (LCEL) |
| 간단한 도구 사용 | Agent (ReAct) |
| 복잡한 워크플로우 | LangGraph |
| 다중 에이전트 시스템 | LangGraph |
| 상태 유지 에이전트 | LangGraph |
| 대화 유지 | RunnableWithMessageHistory |

### LCEL vs LangGraph

| 기능 | LCEL | LangGraph |
|------|------|-----------|
| **사용 사례** | 선형/간단한 분기 | 사이클, 복잡한 라우팅 |
| **상태** | 상태 없음 | 상태 유지 |
| **문법** | 파이프 연산자 (`\|`) | StateGraph |
| **복잡도** | 간단~중간 | 중간~복잡 |
| **최적 용도** | RAG, 간단한 에이전트 | 다중 에이전트, human-in-loop |

---

## 다음 단계

[11_Vector_Databases.md](./11_Vector_Databases.md)에서 벡터 데이터베이스를 학습합니다.
