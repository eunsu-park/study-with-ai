# 12. Practical Chatbot Project

## Learning Objectives

- Designing conversational AI systems
- Implementing RAG-based chatbots
- Conversation management and memory
- Production deployment considerations

---

## 1. Chatbot Architecture

### Basic Structure

```
┌─────────────────────────────────────────────────────────────┐
│                       Chatbot System                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Input                                                   │
│      │                                                       │
│      ▼                                                       │
│  [Intent Classification] ──▶ FAQ / RAG / General dialogue branch │
│      │                                                       │
│      ▼                                                       │
│  [Context Retrieval] ◀── Vector DB                          │
│      │                                                       │
│      ▼                                                       │
│  [Prompt Construction] ◀── Conversation History             │
│      │                                                       │
│      ▼                                                       │
│  [LLM Generation]                                            │
│      │                                                       │
│      ▼                                                       │
│  Response Output                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Basic Chatbot Implementation

### Simple Conversational Chatbot

```python
from openai import OpenAI

class SimpleChatbot:
    def __init__(self, system_prompt=None):
        self.client = OpenAI()
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.history = []

    def chat(self, user_message):
        # Construct messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})

        # API call
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )

        assistant_message = response.choices[0].message.content

        # Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def clear_history(self):
        self.history = []

# Usage
bot = SimpleChatbot("You are a friendly customer support agent.")
print(bot.chat("Hi, I need help with my order."))
print(bot.chat("My order number is 12345."))
```

---

## 3. RAG Chatbot

### Document-Based Q&A

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

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Conversation history
        self.history = []

        # Setup RAG chain
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

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_sources(self, question):
        """Return retrieved source documents"""
        docs = self.retriever.invoke(question)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
```

---

## 4. Advanced Conversation Management

### Intent Classification

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

# Usage
classifier = IntentClassifier()
intent = classifier.classify(
    "I want to return my purchase",
    ["order_status", "return_request", "product_inquiry", "general"]
)
# "return_request"
```

### Slot Extraction

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

# Usage
extractor = SlotExtractor()
slots = extractor.extract("I want to return order #12345, the shirt is too small")
# {'order_id': '12345', 'product_name': 'shirt', 'issue': 'too small'}
```

### Conversation State Management

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
        # Extract slots
        new_slots = self.extract_slots(message)
        self.context.slots.update({k: v for k, v in new_slots.items() if v})

        # Check missing slots
        self.context.missing_slots = [
            s for s in self.required_slots
            if s not in self.context.slots or not self.context.slots[s]
        ]

        # State transition
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

## 5. Streaming Responses

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

        # Update history
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})

# Usage
bot = StreamingChatbot()
for chunk in bot.chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 6. FastAPI Web Server

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Session storage
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Get/create session
    if request.session_id not in sessions:
        sessions[request.session_id] = RAGChatbot(documents)

    bot = sessions[request.session_id]

    # Generate response
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

# Usage
ui = ChatbotUI()
ui.launch()
```

---

## 8. Production Considerations

### Error Handling

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

### Token/Cost Management

```python
import tiktoken

class TokenManager:
    def __init__(self, model="gpt-3.5-turbo", max_tokens=4000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(self.encoding.encode(text))

    def truncate_history(self, history, max_history_tokens=2000):
        """Remove older messages first"""
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

### Monitoring

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

        # Record metrics
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

## Summary

### Chatbot Design Checklist

```
□ Define purpose (general dialogue / FAQ / document-based)
□ Decide if RAG is needed
□ Conversation history management approach
□ Need for intent classification
□ Error handling and fallbacks
□ Cost management (token limits)
□ Monitoring and logging
```

### Core Patterns

```python
# Basic chatbot
messages = [system_prompt] + history + [user_message]
response = llm.invoke(messages)

# RAG chatbot
docs = retriever.invoke(query)
context = format_docs(docs)
response = llm.invoke(prompt.format(context=context, question=query))

# Streaming
for chunk in llm.stream(messages):
    yield chunk
```

### Next Steps

- Deploy to production (AWS, GCP)
- Set up A/B testing
- Collect user feedback
- Continuous model improvement

---

## Course Complete

This completes the LLM & NLP learning course!

### Course Summary

1. **NLP Basics (01-03)**: Tokenization, embeddings, Transformer
2. **Pre-trained Models (04-07)**: BERT, GPT, HuggingFace, fine-tuning
3. **LLM Applications (08-12)**: Prompting, RAG, LangChain, vector DBs, chatbots

### Recommended Next Steps

- Apply to real projects
- Participate in Kaggle NLP competitions
- Read latest LLM papers (Claude, Gemini, Llama)
