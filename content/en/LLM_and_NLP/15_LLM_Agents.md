# 15. LLM Agents

## Learning Objectives

- Understand agent concepts and architecture
- Implement ReAct pattern
- Tool use techniques
- LangChain Agent utilization
- Autonomous agent systems (AutoGPT, etc.)

---

## 1. LLM Agent Overview

### What is an Agent?

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Agent                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                            │
│  │    LLM      │  ◀── Brain (decision-making)               │
│  │  (Brain)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │   Planning  │  ◀── Plan formulation                      │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │    Tools    │    │   Memory    │  ◀── Tools + Memory     │
│  │ (Search,    │    │ (Chat hist, │                         │
│  │  Calculator,│    │  Knowledge  │                         │
│  │  Code exec) │    │     base)   │                         │
│  └─────────────┘    └─────────────┘                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Agent vs Chatbot

| Aspect | Chatbot | Agent |
|--------|---------|-------|
| Response Method | Single response | Multi-step reasoning |
| Tool Use | Limited | Diverse tools |
| Autonomy | Low | High |
| Planning | None | Yes |
| Example | Customer support bot | AutoGPT, Copilot |

---

## 2. ReAct (Reasoning + Acting)

### ReAct Pattern

```
Thought: Analyze problem and decide next action
Action: Select tool and determine input
Observation: Tool execution result
... (repeat)
Final Answer: Final response
```

### ReAct Implementation

```python
from openai import OpenAI

client = OpenAI()

# Tool definitions
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: f"Search results: Information about {query}...",
    "get_weather": lambda city: f"Weather in {city}: Sunny, 25°C",
}

def react_agent(question, max_steps=5):
    """ReAct agent"""

    system_prompt = """You are an agent that solves problems step by step.

Available tools:
- calculator: Perform math calculations (e.g., "2 + 3 * 4")
- search: Search for information (e.g., "Python creator")
- get_weather: Check weather (e.g., "Seoul")

Follow this format:

Thought: [Analyze current situation and plan next action]
Action: [tool name]
Action Input: [tool input]

When you receive tool results:
Observation: [result]

When final answer is ready:
Final Answer: [answer]
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

        # Check for Final Answer
        if "Final Answer:" in assistant_message:
            final_answer = assistant_message.split("Final Answer:")[-1].strip()
            return final_answer

        # Parse Action
        if "Action:" in assistant_message and "Action Input:" in assistant_message:
            action_line = assistant_message.split("Action:")[-1].split("\n")[0].strip()
            input_line = assistant_message.split("Action Input:")[-1].split("\n")[0].strip()

            # Execute tool
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

    return "Maximum steps reached, unable to answer"

# Usage
answer = react_agent("Check the weather in Seoul and convert the temperature from Celsius to Fahrenheit.")
print(f"\nFinal answer: {answer}")
```

---

## 3. Tool Use

### Function Calling (OpenAI)

```python
from openai import OpenAI
import json

client = OpenAI()

# Tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name (e.g., Seoul, Tokyo)"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
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
            "description": "Search for information on the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Tool implementations
def get_weather(city, unit="celsius"):
    # In practice, call API
    weather_data = {
        "Seoul": {"temp": 25, "condition": "Sunny"},
        "Tokyo": {"temp": 28, "condition": "Cloudy"},
    }
    data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
    return json.dumps(data)

def search_web(query):
    return json.dumps({"results": f"Search results for '{query}'..."})

tool_implementations = {
    "get_weather": get_weather,
    "search_web": search_web,
}

def agent_with_tools(user_message):
    """Function Calling agent"""
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Automatically select tool
    )

    assistant_message = response.choices[0].message

    # Check if tool call is needed
    if assistant_message.tool_calls:
        messages.append(assistant_message)

        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute tool
            function_response = tool_implementations[function_name](**function_args)

            # Add result
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response
            })

        # Final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return final_response.choices[0].message.content

    return assistant_message.content

# Usage
result = agent_with_tools("Compare the weather in Seoul and Tokyo.")
print(result)
```

### Code Execution Tool

```python
import subprocess
import tempfile
import os

def execute_python(code):
    """Safely execute Python code"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=10  # Set timeout
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        return {"success": result.returncode == 0, "output": output}
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "Timeout"}
    finally:
        os.unlink(temp_path)

# Code execution tool definition
code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}
```

---

## 4. LangChain Agent

### Basic Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun

# LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Tool definitions
search = DuckDuckGoSearchRun()

@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations. Input: math expression (e.g., '2 + 3 * 4')"""
    try:
        return str(eval(expression))
    except:
        return "Calculation error"

@tool
def get_current_time() -> str:
    """Return current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [
    Tool(name="Search", func=search.run, description="Web search"),
    calculator,
    get_current_time,
]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use tools to answer questions."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({"input": "Tell me the current time and today's major news."})
print(result["output"])
```

### ReAct Agent (LangChain)

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct prompt
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

# Create agent
react_agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Execute
result = agent_executor.invoke({"input": "Search for 2024 US presidential election results and summarize."})
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt (with memory)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# Conversation
agent_executor.invoke({"input": "My name is John."})
agent_executor.invoke({"input": "What did I say my name was?"})
```

---

## 5. Autonomous Agent Systems

### Plan-and-Execute

```python
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Create Planner and Executor
planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

# Plan-and-Execute agent
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# Execute complex task
result = agent.run("Research the history of Python and create a markdown document summarizing key features by major version.")
```

### AutoGPT Style Agent

```python
class AutoGPTAgent:
    """Autonomous agent"""

    def __init__(self, llm, tools, goals):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.goals = goals
        self.memory = []
        self.completed_tasks = []

    def plan(self):
        """Plan to achieve goals"""
        prompt = f"""You are an autonomous AI agent.

Goals: {self.goals}

Completed tasks:
{self.completed_tasks}

Previous task results:
{self.memory[-5:] if self.memory else "None"}

Available tools:
{list(self.tools.keys())}

Output the next task in JSON format:
{{"task": "task description", "tool": "tool to use", "input": "tool input"}}

If all goals are achieved:
{{"task": "COMPLETE", "summary": "result summary"}}
"""
        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def execute(self, task):
        """Execute task"""
        if task["task"] == "COMPLETE":
            return {"status": "complete", "summary": task["summary"]}

        tool = self.tools.get(task["tool"])
        if tool:
            result = tool.run(task["input"])
            return {"status": "success", "result": result}
        return {"status": "error", "message": f"Unknown tool: {task['tool']}"}

    def run(self, max_iterations=10):
        """Run agent"""
        for i in range(max_iterations):
            print(f"\n=== Iteration {i+1} ===")

            # Plan
            task = self.plan()
            print(f"Task: {task}")

            # Check completion
            if task.get("task") == "COMPLETE":
                print(f"Goals achieved: {task['summary']}")
                return task["summary"]

            # Execute
            result = self.execute(task)
            print(f"Result: {result}")

            # Update memory
            self.memory.append({"task": task, "result": result})
            if result["status"] == "success":
                self.completed_tasks.append(task["task"])

        return "Max iterations reached"

# Usage
agent = AutoGPTAgent(
    llm=ChatOpenAI(model="gpt-4"),
    tools=tools,
    goals=["Research Seoul population", "Analyze demographics", "Create report"]
)
result = agent.run()
```

---

## 6. Multi-Agent Systems

### Agent Collaboration

```python
class ResearcherAgent:
    """Research agent"""
    def __init__(self, llm):
        self.llm = llm

    def research(self, topic):
        prompt = f"Research '{topic}' and summarize key information."
        return self.llm.invoke(prompt).content

class WriterAgent:
    """Writing agent"""
    def __init__(self, llm):
        self.llm = llm

    def write(self, research_results, style="formal"):
        prompt = f"Write a {style} style document based on the following information:\n{research_results}"
        return self.llm.invoke(prompt).content

class ReviewerAgent:
    """Review agent"""
    def __init__(self, llm):
        self.llm = llm

    def review(self, document):
        prompt = f"Review the following document and suggest improvements:\n{document}"
        return self.llm.invoke(prompt).content

class MultiAgentSystem:
    """Multi-agent system"""

    def __init__(self, llm):
        self.researcher = ResearcherAgent(llm)
        self.writer = WriterAgent(llm)
        self.reviewer = ReviewerAgent(llm)

    def create_document(self, topic, max_revisions=2):
        # 1. Research
        print("=== Research Phase ===")
        research = self.researcher.research(topic)
        print(research[:200] + "...")

        # 2. Write
        print("\n=== Writing Phase ===")
        document = self.writer.write(research)
        print(document[:200] + "...")

        # 3. Review and revise
        for i in range(max_revisions):
            print(f"\n=== Review {i+1} ===")
            review = self.reviewer.review(document)
            print(review[:200] + "...")

            # Revise
            if "no revisions needed" in review:
                break
            document = self.writer.write(f"Original:\n{document}\n\nReview:\n{review}", style="revised")

        return document

# Usage
llm = ChatOpenAI(model="gpt-4")
system = MultiAgentSystem(llm)
final_doc = system.create_document("The Future of Artificial Intelligence")
```

---

## 7. Agent Evaluation

### Tool Selection Accuracy

```python
def evaluate_tool_selection(agent, test_cases):
    """Evaluate tool selection accuracy"""
    correct = 0
    total = len(test_cases)

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]

        # Run agent (tool selection only)
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

# Test cases
test_cases = [
    {"query": "Calculate 2 + 3 * 4", "expected_tool": "calculator"},
    {"query": "What's the weather in Seoul today?", "expected_tool": "get_weather"},
    {"query": "Who is the creator of Python?", "expected_tool": "search"},
]

# Evaluate
evaluate_tool_selection(agent, test_cases)
```

### Task Completion Rate

```python
def evaluate_task_completion(agent, tasks):
    """Evaluate task completion rate"""
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

# Task definitions
tasks = [
    {
        "description": "Weather check and clothing recommendation",
        "input": "Check Seoul weather and recommend what to wear today",
        "validator": lambda r: "Seoul" in r and ("clothing" in r or "wear" in r)
    },
    {
        "description": "Math calculation",
        "input": "What is 123 * 456?",
        "validator": lambda r: "56088" in r
    },
]
```

---

## Summary

### Agent Architecture Comparison

| Architecture | Features | When to Use |
|--------------|----------|-------------|
| ReAct | Reasoning-action iteration | Step-by-step problem solving |
| Function Calling | Structured tool calls | API integration |
| Plan-and-Execute | Plan then execute | Complex tasks |
| AutoGPT | Autonomous goal achievement | Long-term tasks |
| Multi-Agent | Role-based collaboration | Specialized expertise needed |

### Core Code

```python
# ReAct pattern
Thought: Analyze problem
Action: Select tool
Observation: Check result
Final Answer: Final response

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

### Agent Design Checklist

```
□ Clear tool definitions (name, description, parameters)
□ Error handling (tool failures, parsing errors)
□ Memory management (chat history, context)
□ Loop prevention (maximum iterations)
□ Safety measures (restrict dangerous operations)
□ Logging and monitoring
```

---

## Next Steps

In [16_Evaluation_Metrics.md](./16_Evaluation_Metrics.md), we'll learn about LLM evaluation metrics and benchmarks.
