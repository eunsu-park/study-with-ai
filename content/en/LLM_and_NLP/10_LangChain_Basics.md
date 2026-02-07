# 10. LangChain Basics

## Learning Objectives

- LangChain core concepts
- LLM wrappers and prompts
- Chains and agents
- Memory systems

---

## 1. LangChain Overview

### Installation

```bash
pip install langchain langchain-openai langchain-community
```

### Core Components

```
LangChain
├── Models          # LLM wrappers
├── Prompts         # Prompt templates
├── Chains          # Sequential calls
├── Agents          # Tool-using agents
├── Memory          # Conversation history
├── Retrievers      # Document retrieval
└── Callbacks       # Monitoring
```

---

## 2. LLM Wrappers

### ChatOpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

# Simple call
response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Various LLMs

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

# Ollama (local)
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### Message Types

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

## 3. Prompt Templates

### Basic Template

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}."
)

prompt = template.format(topic="spring")
response = llm.invoke(prompt)
```

### Chat Prompts

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

### Few-shot Prompts

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

## 4. Chains

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Compose chain
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

# Connect with pipe operator
chain = prompt | llm | output_parser

# Execute
result = chain.invoke({"topic": "programmers"})
print(result)
```

### Sequential Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# First chain: generate topic
topic_prompt = ChatPromptTemplate.from_template(
    "Generate a random topic for a story."
)

# Second chain: write story
story_prompt = ChatPromptTemplate.from_template(
    "Write a short story about: {topic}"
)

# Connect chains
chain = (
    {"topic": topic_prompt | llm | StrOutputParser()}
    | story_prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({})
```

### Parallel Chains

```python
from langchain_core.runnables import RunnableParallel

# Parallel execution
parallel_chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Long article here..."})
# {'summary': '...', 'keywords': '...', 'sentiment': '...'}
```

---

## 5. Output Parsers

### String Parser

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = prompt | llm | parser  # Convert to string
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

### Structured Output

```python
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieReview)
```

---

## 6. Agents

### Basic Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun

# Define tools
search = DuckDuckGoSearchRun()
tools = [search]

# Load ReAct prompt
prompt = hub.pull("hwchase17/react")

# Create agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "What is the weather in Seoul?"})
```

### Custom Tools

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

### Tool Class

```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import Field

class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search for information on the internet"

    def _run(self, query: str) -> str:
        # Search logic
        return f"Search results for: {query}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
```

---

## 7. Memory

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Conversation
response1 = conversation.predict(input="Hi, I'm John")
response2 = conversation.predict(input="What's my name?")
# "Your name is John"
```

### Summary Memory

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)

# Summarize and store long conversations
```

### Window Memory

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep only recent k conversations
memory = ConversationBufferWindowMemory(k=5)
```

### Memory in LCEL

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

# Usage
response = chain_with_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "user123"}}
)
```

---

## 8. RAG with LangChain

### Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader
)

# Text file
loader = TextLoader("document.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Web page
loader = WebBaseLoader("https://example.com")
docs = loader.load()
```

### Text Splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)
```

### Vector Stores

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("What is machine learning?")
```

### RAG Chain

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

## 9. Streaming

```python
# Streaming output
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## Summary

### Core Patterns

```python
# Basic chain
chain = prompt | llm | output_parser

# RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | parser
)

# Agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### Component Selection Guide

| Situation | Component |
|-----------|-----------|
| Simple call | LLM + Prompt |
| Sequential processing | Chain (LCEL) |
| Document-based | RAG Chain |
| Tool usage | Agent |
| Maintain conversation | Memory |

---

## Next Steps

Learn about vector databases in [11_Vector_Databases.md](./11_Vector_Databases.md).
