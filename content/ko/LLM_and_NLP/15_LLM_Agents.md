# 15. LLM 에이전트 (LLM Agents)

## 학습 목표

- 에이전트 개념과 아키텍처 이해
- ReAct 패턴 구현
- 도구 사용 (Tool Use) 기법
- LangChain Agent 활용
- 자율 에이전트 시스템 (AutoGPT 등)

---

## 1. LLM 에이전트 개요

### 에이전트란?

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM 에이전트                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                            │
│  │    LLM      │  ◀── 두뇌 (의사결정)                       │
│  │  (Brain)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   Planning  │  ◀── 계획 수립                             │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │    Tools    │    │   Memory    │  ◀── 도구 + 기억        │
│  │ (검색, 계산, │    │ (대화 이력, │                         │
│  │  코드실행)   │    │  지식 베이스)│                         │
│  └─────────────┘    └─────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 에이전트 vs 챗봇

| 항목 | 챗봇 | 에이전트 |
|------|------|----------|
| 응답 방식 | 단일 응답 | 다단계 추론 |
| 도구 사용 | 제한적 | 다양한 도구 |
| 자율성 | 낮음 | 높음 |
| 계획 수립 | 없음 | 있음 |
| 예시 | 고객 지원 봇 | AutoGPT, Copilot |

---

## 2. ReAct (Reasoning + Acting)

### ReAct 패턴

```
Thought: 문제를 분석하고 다음 행동 결정
Action: 도구 선택 및 입력 결정
Observation: 도구 실행 결과
... (반복)
Final Answer: 최종 답변
```

### ReAct 구현

```python
from openai import OpenAI

client = OpenAI()

# 도구 정의
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: f"검색 결과: {query}에 대한 정보...",
    "get_weather": lambda city: f"{city}의 날씨: 맑음, 25도",
}

def react_agent(question, max_steps=5):
    """ReAct 에이전트"""

    system_prompt = """당신은 문제를 단계별로 해결하는 에이전트입니다.

사용 가능한 도구:
- calculator: 수학 계산 (예: "2 + 3 * 4")
- search: 정보 검색 (예: "파이썬 창시자")
- get_weather: 날씨 조회 (예: "서울")

다음 형식을 따르세요:

Thought: [현재 상황 분석 및 다음 행동 계획]
Action: [도구 이름]
Action Input: [도구 입력]

도구 결과를 받으면:
Observation: [결과]

최종 답변이 준비되면:
Final Answer: [답변]
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0
        )

        assistant_message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_message})

        print(f"=== Step {step + 1} ===")
        print(assistant_message)

        # Final Answer 체크
        if "Final Answer:" in assistant_message:
            final_answer = assistant_message.split("Final Answer:")[-1].strip()
            return final_answer

        # Action 파싱
        if "Action:" in assistant_message and "Action Input:" in assistant_message:
            action_line = assistant_message.split("Action:")[-1].split("\n")[0].strip()
            input_line = assistant_message.split("Action Input:")[-1].split("\n")[0].strip()

            # 도구 실행
            if action_line in tools:
                try:
                    observation = tools[action_line](input_line)
                except Exception as e:
                    observation = f"Error: {str(e)}"

                observation_message = f"Observation: {observation}"
                messages.append({"role": "user", "content": observation_message})
                print(observation_message)
            else:
                messages.append({"role": "user", "content": f"Error: Unknown tool '{action_line}'"})

    return "최대 단계 도달, 답변 실패"

# 사용
answer = react_agent("서울의 날씨를 확인하고, 기온을 섭씨에서 화씨로 변환해주세요.")
print(f"\n최종 답변: {answer}")
```

---

## 3. 도구 사용 (Tool Use)

### Function Calling (OpenAI)

```python
from openai import OpenAI
import json

client = OpenAI()

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "특정 도시의 현재 날씨 정보를 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "도시 이름 (예: Seoul, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "온도 단위"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "웹에서 정보를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색어"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 도구 구현
def get_weather(city, unit="celsius"):
    # 실제로는 API 호출
    weather_data = {
        "Seoul": {"temp": 25, "condition": "Sunny"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return json.dumps(data)

def search_web(query):
    return json.dumps({"results": f"'{query}'에 대한 검색 결과..."})

tool_implementations = {
    "get_weather": get_weather,
    "search_web": search_web,
}

def agent_with_tools(user_message):
    """Function Calling 에이전트"""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 자동으로 도구 선택
    )

    assistant_message = response.choices[0].message

    # 도구 호출 필요 여부 확인
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # 각 도구 호출 처리
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # 도구 실행
            function_response = tool_implementations[function_name](**function_args)

            # 결과 추가
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

        # 최종 응답
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content

    return assistant_message.content

# 사용
result = agent_with_tools("서울과 도쿄의 날씨를 비교해주세요.")
print(result)
```

### 코드 실행 도구

```python
import subprocess
import tempfile
import os

def execute_python(code):
    """Python 코드 안전하게 실행"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10  # 타임아웃 설정
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {"success": result.returncode == 0, "output": output}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "Timeout"}
    finally:
        os.unlink(temp_path)

# 코드 실행 도구 정의
code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Python 코드를 실행합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "실행할 Python 코드"
                }
            },
            "required": ["code"]
        }
    }
}
```

---

## 4. LangChain Agent

### 기본 에이전트

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 도구 정의
search = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """수학 계산을 수행합니다. 입력: 수학 표현식 (예: '2 + 3 * 4')"""
    try:
        return str(eval(expression))
    except:
        return "계산 오류"

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(name="Search", func=search.run, description="웹 검색"),
    calculator,
    get_current_time,
]

# 프롬프트
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 어시스턴트입니다. 도구를 사용하여 질문에 답하세요."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 에이전트 생성
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 실행
result = agent_executor.invoke({"input": "현재 시간과 오늘의 주요 뉴스를 알려주세요."})
print(result["output"])
```

### ReAct Agent (LangChain)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct 프롬프트
react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

# 에이전트 생성
react_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 실행
result = agent_executor.invoke({"input": "2024년 미국 대통령 선거 결과를 검색하고 요약해주세요."})
```

### 메모리가 있는 에이전트

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 메모리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 프롬프트 (메모리 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 AI 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 에이전트
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# 대화
agent_executor.invoke({"input": "내 이름은 김철수야."})
agent_executor.invoke({"input": "내 이름이 뭐라고 했지?"})
```

---

## 5. 자율 에이전트 시스템

### Plan-and-Execute

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Planner와 Executor 생성
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# Plan-and-Execute 에이전트
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# 복잡한 작업 실행
result = agent.run("파이썬의 역사에 대해 조사하고, 주요 버전별 특징을 요약한 마크다운 문서를 작성해주세요.")
```

### AutoGPT 스타일 에이전트

```python
class AutoGPTAgent:
    """자율 에이전트"""

    def __init__(self, llm, tools, goals):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self):
        """목표 달성을 위한 계획 수립"""
        prompt = f"""당신은 자율 AI 에이전트입니다.

목표: {self.goals}

완료된 작업:
{self.completed_tasks}

이전 작업 결과:
{self.memory[-5:] if self.memory else "없음"}

사용 가능한 도구:
{list(self.tools.keys())}

다음 작업을 JSON 형식으로 출력하세요:
{{"task": "작업 설명", "tool": "사용할 도구", "input": "도구 입력"}}

모든 목표가 달성되었다면:
{{"task": "COMPLETE", "summary": "결과 요약"}}
"""
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def execute(self, task):
        """작업 실행"""
        if task["task"] == "COMPLETE":
            return {"status": "complete", "summary": task["summary"]}

        tool = self.tools.get(task["tool"])
        if tool:
            result = tool.run(task["input"])
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown tool: {task['tool']}"}

    def run(self, max_iterations=10):
        """에이전트 실행"""
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")

            # 계획
            task = self.plan()
            print(f"Task: {task}")

            # 완료 확인
            if task.get("task") == "COMPLETE":
                print(f"Goals achieved: {task['summary']}")
                return task["summary"]

            # 실행
            result = self.execute(task)
            print(f"Result: {result}")

            # 메모리 업데이트
            self.memory.append({"task": task, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(task["task"])

        return "Max iterations reached"

# 사용
agent = AutoGPTAgent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    goals=["서울의 인구 조사", "인구 통계 분석", "보고서 작성"]
)
result = agent.run()
```

---

## 6. 멀티 에이전트 시스템

### 에이전트 간 협업

```python
class ResearcherAgent:
    """연구 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def research(self, topic):
        prompt = f"'{topic}'에 대해 조사하고 핵심 정보를 정리해주세요."
        return self.llm.invoke(prompt).content

class WriterAgent:
    """작문 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def write(self, research_results, style="formal"):
        prompt = f"다음 정보를 바탕으로 {style} 스타일의 문서를 작성해주세요:\n{research_results}"
        return self.llm.invoke(prompt).content

class ReviewerAgent:
    """검토 에이전트"""
    def __init__(self, llm):
        self.llm = llm

    def review(self, document):
        prompt = f"다음 문서를 검토하고 개선점을 제안해주세요:\n{document}"
        return self.llm.invoke(prompt).content

class MultiAgentSystem:
    """멀티 에이전트 시스템"""

    def __init__(self, llm):
        self.researcher = ResearcherAgent(llm)
        self.writer = WriterAgent(llm)
        self.reviewer = ReviewerAgent(llm)

    def create_document(self, topic, max_revisions=2):
        # 1. 연구
        print("=== 연구 단계 ===")
        research = self.researcher.research(topic)
        print(research[:200] + "...")

        # 2. 작성
        print("\n=== 작성 단계 ===")
        document = self.writer.write(research)
        print(document[:200] + "...")

        # 3. 검토 및 수정
        for i in range(max_revisions):
            print(f"\n=== 검토 {i+1} ===")
            review = self.reviewer.review(document)
            print(review[:200] + "...")

            # 수정
            if "수정 필요 없음" in review:
                break
            document = self.writer.write(f"원본:\n{document}\n\n검토:\n{review}", style="revised")

        return document

# 사용
llm = ChatOpenAI(model="gpt-4")
system = MultiAgentSystem(llm)
final_doc = system.create_document("인공지능의 미래")
```

---

## 7. 에이전트 평가

### 도구 선택 정확도

```python
def evaluate_tool_selection(agent, test_cases):
    """도구 선택 정확도 평가"""
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]

        # 에이전트 실행 (도구 선택만)
        result = agent.plan(query)
        selected_tool = result.get("tool")

        if selected_tool == expected_tool:
            correct += 1
            print(f"[CORRECT] Query: {query}, Tool: {selected_tool}")
        else:
            print(f"[WRONG] Query: {query}, Expected: {expected_tool}, Got: {selected_tool}")

    accuracy = correct / total
    print(f"\nTool Selection Accuracy: {accuracy:.2%}")
    return accuracy

# 테스트 케이스
test_cases = [
    {"query": "2 + 3 * 4를 계산해줘", "expected_tool": "calculator"},
    {"query": "오늘 서울 날씨 어때?", "expected_tool": "get_weather"},
    {"query": "파이썬 창시자가 누구야?", "expected_tool": "search"},
]

# 평가
evaluate_tool_selection(agent, test_cases)
```

### 작업 완료율

```python
def evaluate_task_completion(agent, tasks):
    """작업 완료율 평가"""
    results = []

    for task in tasks:
        try:
            result = agent.run(task["input"])
            success = task["validator"](result)
            results.append({
                "task": task["description"],
                "success": success,
                "result": result
            })
        except Exception as e:
            results.append({
                "task": task["description"],
                "success": False,
                "error": str(e)
            })

    completion_rate = sum(r["success"] for r in results) / len(results)
    print(f"Task Completion Rate: {completion_rate:.2%}")
    return results

# 작업 정의
tasks = [
    {
        "description": "날씨 조회 및 옷차림 추천",
        "input": "서울 날씨를 확인하고 오늘 옷차림을 추천해줘",
        "validator": lambda r: "서울" in r and ("옷" in r or "의류" in r)
    },
    {
        "description": "수학 계산",
        "input": "123 * 456의 결과는?",
        "validator": lambda r: "56088" in r
    },
]
```

---

## 정리

### 에이전트 아키텍처 비교

| 아키텍처 | 특징 | 사용 시점 |
|----------|------|----------|
| ReAct | 추론-행동 반복 | 단계별 문제 해결 |
| Function Calling | 구조화된 도구 호출 | API 연동 |
| Plan-and-Execute | 계획 후 실행 | 복잡한 작업 |
| AutoGPT | 자율 목표 달성 | 장기 작업 |
| Multi-Agent | 역할 분담 협업 | 전문성 필요 |

### 핵심 코드

```python
# ReAct 패턴
Thought: 문제 분석
Action: 도구 선택
Observation: 결과 확인
Final Answer: 최종 답변

# Function Calling (OpenAI)
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# LangChain Agent
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": query})
```

### 에이전트 설계 체크리스트

```
□ 명확한 도구 정의 (이름, 설명, 파라미터)
□ 에러 처리 (도구 실패, 파싱 오류)
□ 메모리 관리 (대화 이력, 컨텍스트)
□ 루프 방지 (최대 반복 횟수)
□ 안전 장치 (위험한 작업 제한)
□ 로깅 및 모니터링
```

---

## 다음 단계

[16_Evaluation_Metrics.md](./16_Evaluation_Metrics.md)에서 LLM 평가 지표와 벤치마크를 학습합니다.
