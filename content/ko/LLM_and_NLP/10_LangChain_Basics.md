# 10. LangChain 기초

## 학습 목표

- LangChain 핵심 개념
- LLM 래퍼와 프롬프트
- 체인과 에이전트
- 메모리 시스템

---

## 1. LangChain 개요

### 설치

```bash
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

### 대화 버퍼 메모리

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

### 요약 메모리

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# 긴 대화를 요약하여 저장
```

### 윈도우 메모리

```python
from langchain.memory import ConversationBufferWindowMemory

# 최근 k개의 대화만 유지
memory = ConversationBufferWindowMemory(k=5)
```

### LCEL에서 메모리

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

## 정리

### 핵심 패턴

```python
# 기본 체인
chain = prompt | llm | output_parser

# RAG 체인
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | parser
)

# 에이전트
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### 컴포넌트 선택 가이드

| 상황 | 컴포넌트 |
|------|----------|
| 단순 호출 | LLM + Prompt |
| 순차 처리 | Chain (LCEL) |
| 문서 기반 | RAG Chain |
| 도구 사용 | Agent |
| 대화 유지 | Memory |

---

## 다음 단계

[11_Vector_Databases.md](./11_Vector_Databases.md)에서 벡터 데이터베이스를 학습합니다.
