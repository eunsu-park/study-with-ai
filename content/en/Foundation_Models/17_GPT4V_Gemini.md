# 17. GPT-4V, GPT-4o, Gemini & Claude 3

## Overview

GPT-4V(ision), GPT-4o, Gemini, and Claude 3 are currently the most powerful commercial multimodal AI systems. This lesson covers their features, API usage, and practical applications.

> **2024 Updates**:
> - **GPT-4o** (May 2024): "omni" version of GPT-4, native multimodal
> - **Gemini 1.5 Pro**: 2M token context, native video/audio
> - **Claude 3 Family** (March 2024): Haiku, Sonnet, Opus lineup
> - **Claude 3.5 Sonnet** (June 2024): Enhanced vision capabilities

---

## 1. GPT-4V (GPT-4 with Vision)

### 1.1 Feature Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4V Key Features                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🖼️ Image Understanding                                          │
│  - Detailed description and analysis                            │
│  - Multi-image comparison                                       │
│  - Chart/graph interpretation                                   │
│                                                                  │
│  📝 Text Recognition (OCR)                                       │
│  - Handwriting recognition                                      │
│  - Multilingual text                                            │
│  - Document structure understanding                             │
│                                                                  │
│  🔍 Detailed Analysis                                            │
│  - Object identification and counting                           │
│  - Spatial relationship understanding                           │
│  - Attribute reasoning                                          │
│                                                                  │
│  💡 Reasoning and Creation                                       │
│  - Image-based reasoning                                        │
│  - Code generation (UI screenshot → code)                       │
│  - Creative writing                                             │
│                                                                  │
│  ⚠️ Limitations                                                  │
│  - No medical diagnosis                                         │
│  - No face recognition/identity verification                    │
│  - No real-time video (images only)                             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 API Usage

```python
from openai import OpenAI
import base64
from pathlib import Path

client = OpenAI()

def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def gpt4v_basic(image_path: str, prompt: str) -> str:
    """Basic image analysis"""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4v_multi_image(image_paths: list, prompt: str) -> str:
    """Multi-image analysis"""

    content = [{"type": "text", "text": prompt}]

    for path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
        })

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{"role": "user", "content": content}],
        max_tokens=2048
    )

    return response.choices[0].message.content


def gpt4v_with_detail(image_path: str, prompt: str, detail: str = "high") -> str:
    """
    Specify detail level

    detail:
    - "low": fast and cheap, low-resolution analysis
    - "high": detailed analysis, more tokens used
    - "auto": automatic selection
    """

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4v_url_image(image_url: str, prompt: str) -> str:
    """Analyze image from URL"""

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content
```

### 1.3 Practical Applications

```python
class GPT4VApplications:
    """GPT-4V practical applications"""

    def __init__(self):
        self.client = OpenAI()

    def analyze_ui_screenshot(self, screenshot_path: str) -> dict:
        """UI screenshot analysis and code generation"""

        prompt = """Analyze this UI screenshot and:
        1. List all UI components visible
        2. Describe the layout structure
        3. Generate HTML/CSS code to recreate this UI

        Format your response as JSON with keys:
        - components: list of UI elements
        - layout: description of layout
        - html_code: HTML implementation
        - css_code: CSS styles
        """

        response = self._call_api(screenshot_path, prompt)

        # Parse JSON
        import json
        try:
            return json.loads(response)
        except:
            return {"raw_response": response}

    def extract_data_from_chart(self, chart_path: str) -> dict:
        """Extract data from charts"""

        prompt = """Analyze this chart and extract:
        1. Chart type (bar, line, pie, etc.)
        2. Title and axis labels
        3. All data points with their values
        4. Key insights or trends

        Return as structured JSON.
        """

        return self._call_api(chart_path, prompt)

    def compare_images(self, image_paths: list) -> str:
        """Image comparison analysis"""

        prompt = """Compare these images and describe:
        1. Similarities
        2. Differences
        3. Which image is better quality and why
        4. Any notable features in each
        """

        return gpt4v_multi_image(image_paths, prompt)

    def ocr_with_structure(self, document_path: str) -> dict:
        """Structured OCR"""

        prompt = """Extract all text from this document and preserve:
        1. Headings and hierarchy
        2. Tables (as markdown)
        3. Lists (numbered and bulleted)
        4. Key-value pairs

        Return as structured markdown.
        """

        return self._call_api(document_path, prompt)

    def generate_alt_text(self, image_path: str) -> str:
        """Generate alt text for web accessibility"""

        prompt = """Generate an appropriate alt text for this image.
        The alt text should be:
        1. Concise (under 125 characters)
        2. Descriptive of the main content
        3. Useful for screen reader users

        Just return the alt text, nothing else.
        """

        return self._call_api(image_path, prompt)

    def _call_api(self, image_path: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048
        )
        return response.choices[0].message.content
```

---

## 2. GPT-4o (Omni)

### 2.1 GPT-4o Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    GPT-4o vs GPT-4V Comparison                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPT-4V (Previous):                                              │
│  - Text + image input                                            │
│  - Separate vision encoder                                       │
│  - Relatively slower response                                    │
│                                                                  │
│  GPT-4o (May 2024):                                              │
│  - Text + image + audio native                                   │
│  - Single model handles all modalities                           │
│  - 2x faster response, 50% cheaper                               │
│  - Real-time voice conversation                                  │
│                                                                  │
│  Key Improvements:                                                │
│  ✅ Speed: Average 320ms response (2x faster than GPT-4V)        │
│  ✅ Cost: $5/1M input, $15/1M output                             │
│  ✅ Vision: Improved OCR, chart interpretation                   │
│  ✅ Audio: Real-time voice input/output                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 GPT-4o API Usage

```python
from openai import OpenAI
import base64

client = OpenAI()

def gpt4o_vision(image_path: str, prompt: str) -> str:
    """GPT-4o image analysis"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4o
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    return response.choices[0].message.content


def gpt4o_audio(audio_path: str, prompt: str) -> str:
    """GPT-4o audio analysis (Realtime API)"""

    # Read audio file
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": "wav"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


# GPT-4o-mini: Low-cost version
def gpt4o_mini_vision(image_path: str, prompt: str) -> str:
    """GPT-4o-mini: Fast and cheap vision model"""

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Low-cost version
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }
        ],
        max_tokens=512
    )

    return response.choices[0].message.content
```

---

## 3. Google Gemini

### 3.1 Gemini Model Lineup

```
┌──────────────────────────────────────────────────────────────────┐
│                    Gemini Model Comparison                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Gemini 1.5 Flash:                                              │
│  - Fast response, low cost                                      │
│  - 1M token context                                             │
│  - Suitable for real-time applications                          │
│                                                                  │
│  Gemini 1.5 Pro:                                                │
│  - Best performance                                             │
│  - 2M token context                                             │
│  - Complex reasoning, code generation                           │
│                                                                  │
│  Gemini 1.0 Ultra:                                              │
│  - Largest model                                                │
│  - Complex multimodal tasks                                     │
│                                                                  │
│  Special Features:                                               │
│  - Native multimodal (text, image, audio, video)               │
│  - Ultra-long context (1 hour video analysis)                  │
│  - Built-in code execution                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 Gemini API Usage

```python
import google.generativeai as genai
from PIL import Image

# Configure API key
genai.configure(api_key="YOUR_API_KEY")

def gemini_basic(image_path: str, prompt: str) -> str:
    """Basic image analysis"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    image = Image.open(image_path)

    response = model.generate_content([prompt, image])

    return response.text


def gemini_multi_image(image_paths: list, prompt: str) -> str:
    """Multi-image analysis"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    content = [prompt]
    for path in image_paths:
        content.append(Image.open(path))

    response = model.generate_content(content)

    return response.text


def gemini_video_analysis(video_path: str, prompt: str) -> str:
    """Video analysis (Gemini specialized feature)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Upload video
    video_file = genai.upload_file(video_path)

    # Wait for processing
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed")

    response = model.generate_content([prompt, video_file])

    return response.text


def gemini_long_context(documents: list, query: str) -> str:
    """Long document analysis (1M+ tokens)"""

    model = genai.GenerativeModel('gemini-1.5-pro')

    # Combine all documents
    content = [query]
    for doc in documents:
        if doc.endswith('.pdf'):
            content.append(genai.upload_file(doc))
        elif doc.endswith(('.jpg', '.png')):
            content.append(Image.open(doc))
        else:
            with open(doc, 'r') as f:
                content.append(f.read())

    response = model.generate_content(content)

    return response.text


def gemini_with_code_execution(prompt: str) -> dict:
    """Code execution feature"""

    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        tools='code_execution'
    )

    response = model.generate_content(prompt)

    # Extract executed code and results
    result = {
        'text': response.text,
        'code_execution': []
    }

    for part in response.parts:
        if hasattr(part, 'code_execution_result'):
            result['code_execution'].append({
                'code': part.text,
                'output': part.code_execution_result.output
            })

    return result
```

### 3.3 Gemini Specialized Applications

```python
class GeminiApplications:
    """Gemini specialized applications"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    def analyze_long_video(
        self,
        video_path: str,
        questions: list
    ) -> dict:
        """Long video analysis (1 hour+)"""

        video_file = self._upload_and_wait(video_path)

        results = {}

        for question in questions:
            prompt = f"""Analyze this video and answer: {question}

            Provide timestamps when relevant.
            """

            response = self.model.generate_content([prompt, video_file])
            results[question] = response.text

        return results

    def multimodal_reasoning(
        self,
        images: list,
        audio_path: str = None,
        text: str = None
    ) -> str:
        """Multimodal reasoning"""

        content = []

        if text:
            content.append(text)

        for img_path in images:
            content.append(Image.open(img_path))

        if audio_path:
            audio_file = self._upload_and_wait(audio_path)
            content.append(audio_file)

        response = self.model.generate_content(content)

        return response.text

    def research_assistant(
        self,
        pdf_paths: list,
        research_question: str
    ) -> dict:
        """Research assistant (long document analysis)"""

        # Upload PDFs
        files = [self._upload_and_wait(path) for path in pdf_paths]

        prompt = f"""You are a research assistant. Analyze these academic papers
        and answer the following research question:

        {research_question}

        Structure your response as:
        1. Summary of relevant findings from each paper
        2. Synthesis of the findings
        3. Gaps or contradictions
        4. Suggested future directions
        """

        content = [prompt] + files

        response = self.model.generate_content(content)

        return {
            'answer': response.text,
            'sources': pdf_paths
        }

    def _upload_and_wait(self, file_path: str):
        """Upload file and wait for processing"""
        import time

        file = genai.upload_file(file_path)

        while file.state.name == "PROCESSING":
            time.sleep(5)
            file = genai.get_file(file.name)

        return file
```

---

## 4. Anthropic Claude 3

### 4.1 Claude 3 Model Lineup

```
┌──────────────────────────────────────────────────────────────────┐
│                    Claude 3 Family (March 2024)                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Claude 3 Haiku:                                                 │
│  - Fastest and cheapest                                          │
│  - Real-time applications, high-volume processing                │
│  - Vision support                                                │
│                                                                  │
│  Claude 3 Sonnet:                                                │
│  - Balance of speed and performance                              │
│  - Suitable for most business use cases                          │
│  - Vision support                                                │
│                                                                  │
│  Claude 3 Opus:                                                  │
│  - Highest performance                                           │
│  - Complex reasoning, analysis tasks                             │
│  - Vision support                                                │
│                                                                  │
│  Claude 3.5 Sonnet (June 2024):                                  │
│  - Opus-level performance at Sonnet pricing                      │
│  - Enhanced vision, coding capabilities                          │
│  - 200K token context                                            │
│                                                                  │
│  Features:                                                        │
│  ✅ 200K context window (all models)                              │
│  ✅ Multimodal: Image understanding                               │
│  ✅ Safety: Constitutional AI applied                             │
│  ✅ Tool use: Function Calling support                            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Claude API Usage

```python
import anthropic
import base64

client = anthropic.Anthropic()


def claude_vision(image_path: str, prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Claude vision analysis"""

    # Encode image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    if image_path.endswith(".png"):
        media_type = "image/png"
    elif image_path.endswith(".gif"):
        media_type = "image/gif"
    elif image_path.endswith(".webp"):
        media_type = "image/webp"
    else:
        media_type = "image/jpeg"

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text


def claude_multi_image(image_paths: list, prompt: str) -> str:
    """Claude multi-image analysis"""

    content = []

    for path in image_paths:
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        media_type = "image/png" if path.endswith(".png") else "image/jpeg"

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            }
        })

    content.append({"type": "text", "text": prompt})

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}],
    )

    return message.content[0].text


def claude_with_tools(prompt: str, image_path: str = None) -> dict:
    """Claude Tool Use (Function Calling)"""

    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    content = [{"type": "text", "text": prompt}]

    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.insert(0, {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            }
        })

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": content}],
    )

    return {
        "content": message.content,
        "stop_reason": message.stop_reason
    }
```

### 4.3 Claude Specialized Features

```python
class ClaudeApplications:
    """Claude specialized applications"""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def long_document_analysis(self, document_text: str, query: str) -> str:
        """Long document analysis (200K tokens)"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"""Analyze the following document and answer the question.

Document:
{document_text}

Question: {query}
"""
                }
            ],
        )

        return message.content[0].text

    def code_review(self, code: str, language: str = "python") -> str:
        """Code review"""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""Please review the following {language} code.

```{language}
{code}
```

Include:
1. Potential bugs
2. Performance improvements
3. Code style suggestions
4. Security issues
"""
                }
            ],
        )

        return message.content[0].text

    def structured_output(self, image_path: str, schema: dict) -> dict:
        """Generate structured output"""
        import json

        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Analyze this image and return results matching the following JSON schema:

{json.dumps(schema, indent=2)}

Return only JSON."""
                        }
                    ]
                }
            ],
        )

        return json.loads(message.content[0].text)
```

---

## 5. Comparison and Selection Guide

### 5.1 Multimodal Model Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2024 Multimodal Model Comparison                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Feature            GPT-4o      Gemini 1.5 Pro   Claude 3.5 Sonnet          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Image Understanding ★★★★★     ★★★★★         ★★★★★                  │
│  Video Analysis      ✗           ★★★★★ (native) ✗                        │
│  Audio Analysis      ★★★★☆     ★★★★☆         ✗                        │
│  Context             128K        2M               200K                       │
│  Code Execution      ✗           ★★★★☆ (built-in) ✗                       │
│  Speed               ★★★★★     ★★★★☆ (Flash)  ★★★★☆                  │
│  Price               Medium      Low              Medium                     │
│  Coding Ability      ★★★★☆     ★★★★☆         ★★★★★                  │
│  Reasoning Ability   ★★★★★     ★★★★☆         ★★★★★                  │
│                                                                             │
│  Recommended Use Cases:                                                      │
│  - GPT-4o: Real-time multimodal, voice chat, fast response needed           │
│  - Gemini: Video analysis, ultra-long docs, multimodal complex tasks        │
│  - Claude: Complex reasoning, code review, long doc analysis, safety-critical│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Use Case Selection

```python
def select_model(use_case: str) -> str:
    """Select model by use case (2024 update)"""

    recommendations = {
        # GPT-4o is better for
        "ui_to_code": "gpt-4o",
        "realtime_chat": "gpt-4o",
        "voice_assistant": "gpt-4o-audio-preview",
        "quick_vision": "gpt-4o",

        # Gemini is better for
        "video_analysis": "gemini-1.5-pro",
        "very_long_document": "gemini-1.5-pro",  # 2M context
        "audio_transcription": "gemini-1.5-pro",
        "multimodal_app": "gemini-1.5-pro",

        # Claude is better for
        "complex_reasoning": "claude-sonnet-4-20250514",
        "code_review": "claude-sonnet-4-20250514",
        "long_document": "claude-sonnet-4-20250514",  # 200K context
        "safety_critical": "claude-sonnet-4-20250514",

        # Cost optimization
        "high_volume": "gemini-1.5-flash",
        "quick_caption": "gpt-4o-mini",
        "simple_classification": "claude-3-haiku-20240307",
    }

    return recommendations.get(use_case, "gpt-4o")
```

---

## 6. Cost Optimization

### 6.1 Cost Calculation

```python
class CostEstimator:
    """API cost estimation"""

    # 2024 pricing (USD per 1M tokens)
    PRICING = {
        "gpt-4-vision-preview": {
            "input": 10.0,   # per 1M tokens
            "output": 30.0,  # per 1M tokens
            "image_low": 85,   # tokens
            "image_high": 765, # tokens (base) + tiles
        },
        "gpt-4o": {
            "input": 5.0,    # per 1M tokens
            "output": 15.0,  # per 1M tokens
            "image_low": 85,
            "image_high": 765,
        },
        "gpt-4o-mini": {
            "input": 0.15,   # per 1M tokens
            "output": 0.60,  # per 1M tokens
            "image_low": 85,
            "image_high": 765,
        },
        "gemini-1.5-pro": {
            "input": 1.25,   # per 1M tokens
            "output": 5.0,
            "image": 258,  # tokens per image
            "video": 263,  # tokens per second
            "audio": 32,   # tokens per second
        },
        "gemini-1.5-flash": {
            "input": 0.075,
            "output": 0.30,
        },
        "claude-3-opus": {
            "input": 15.0,   # per 1M tokens
            "output": 75.0,
        },
        "claude-sonnet-4-20250514": {
            "input": 3.0,    # per 1M tokens
            "output": 15.0,
        },
        "claude-3-haiku": {
            "input": 0.25,   # per 1M tokens
            "output": 1.25,
        },
    }

    def estimate_gpt4v_cost(
        self,
        num_images: int,
        avg_prompt_tokens: int,
        avg_response_tokens: int,
        detail: str = "high"
    ) -> float:
        """Estimate GPT-4V cost"""

        pricing = self.PRICING["gpt-4-vision-preview"]

        # Image tokens
        if detail == "low":
            image_tokens = num_images * pricing["image_low"]
        else:
            image_tokens = num_images * pricing["image_high"]

        total_input = avg_prompt_tokens + image_tokens
        total_output = avg_response_tokens

        cost = (total_input / 1000 * pricing["input"] +
                total_output / 1000 * pricing["output"])

        return cost

    def estimate_gemini_cost(
        self,
        num_images: int = 0,
        video_seconds: int = 0,
        audio_seconds: int = 0,
        text_chars: int = 0,
        output_chars: int = 0,
        model: str = "gemini-1.5-pro"
    ) -> float:
        """Estimate Gemini cost"""

        pricing = self.PRICING[model]

        input_cost = text_chars / 1000 * pricing["input"]
        output_cost = output_chars / 1000 * pricing["output"]

        if model == "gemini-1.5-pro":
            # Multimedia cost
            image_tokens = num_images * pricing["image"]
            video_tokens = video_seconds * pricing["video"]
            audio_tokens = audio_seconds * pricing["audio"]

            media_chars = (image_tokens + video_tokens + audio_tokens) * 4  # Token → char approximation
            input_cost += media_chars / 1000 * pricing["input"]

        return input_cost + output_cost


# Usage example
estimator = CostEstimator()

# Compare cost for 100 image analysis
gpt4v_cost = estimator.estimate_gpt4v_cost(
    num_images=100,
    avg_prompt_tokens=100,
    avg_response_tokens=500,
    detail="high"
)

gemini_cost = estimator.estimate_gemini_cost(
    num_images=100,
    text_chars=500,
    output_chars=2000,
    model="gemini-1.5-pro"
)

print(f"GPT-4V cost: ${gpt4v_cost:.2f}")
print(f"Gemini Pro cost: ${gemini_cost:.2f}")
```

---

## References

### Official Documentation
- [OpenAI GPT-4o Documentation](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

### Benchmarks
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [VQA Challenge](https://visualqa.org/)
- [LMSYS Chatbot Arena](https://chat.lmsys.org/)

### Related Lessons
- [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md)
- [24_API_Evaluation.md](24_API_Evaluation.md)
