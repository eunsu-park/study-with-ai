# 17. GPT-4V & Gemini

## Overview

GPT-4V(ision) and Gemini are currently the most powerful commercial multimodal AI systems. This lesson covers their features, API usage, and practical applications.

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

## 2. Google Gemini

### 2.1 Gemini Model Lineup

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

### 2.2 Gemini API Usage

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

### 2.3 Gemini Specialized Applications

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

## 3. Comparison and Selection Guide

### 3.1 GPT-4V vs Gemini Comparison

```
┌────────────────────────────────────────────────────────────────┐
│              GPT-4V vs Gemini Comparison                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Feature           GPT-4V              Gemini 1.5 Pro          │
│  ───────────────────────────────────────────────────────────  │
│  Image Understanding ★★★★★            ★★★★★                │
│  Video Analysis      ✗ (frames only)   ★★★★★ (native)       │
│  Audio Analysis      ✗                  ★★★★☆                │
│  Context Length      128K               2M                     │
│  Code Execution      ✗                  ★★★★☆ (built-in)    │
│  Price               High               Medium                 │
│  Speed               Medium             Flash is fast          │
│  Consistency         High               High                   │
│                                                                │
│  Recommended Use Cases:                                        │
│  - GPT-4V: Complex image reasoning, UI code gen, precise OCR  │
│  - Gemini: Video analysis, ultra-long docs, multimodal tasks  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Use Case Selection

```python
def select_model(use_case: str) -> str:
    """Select model by use case"""

    recommendations = {
        # GPT-4V is better for
        "ui_to_code": "gpt-4-vision-preview",
        "precise_ocr": "gpt-4-vision-preview",
        "image_reasoning": "gpt-4-vision-preview",
        "chart_analysis": "gpt-4-vision-preview",

        # Gemini is better for
        "video_analysis": "gemini-1.5-pro",
        "long_document": "gemini-1.5-pro",
        "audio_transcription": "gemini-1.5-pro",
        "multimodal_app": "gemini-1.5-pro",

        # Cost optimization
        "high_volume": "gemini-1.5-flash",
        "quick_caption": "gemini-1.5-flash",
    }

    return recommendations.get(use_case, "gpt-4-vision-preview")
```

---

## 4. Cost Optimization

### 4.1 Cost Calculation

```python
class CostEstimator:
    """API cost estimation"""

    # 2024 pricing (USD)
    PRICING = {
        "gpt-4-vision-preview": {
            "input": 0.01,   # per 1K tokens
            "output": 0.03,  # per 1K tokens
            "image_low": 85,   # tokens
            "image_high": 765, # tokens (base) + tiles
        },
        "gemini-1.5-pro": {
            "input": 0.00125,  # per 1K chars
            "output": 0.00375,
            "image": 258,  # tokens per image
            "video": 263,  # tokens per second
            "audio": 32,   # tokens per second
        },
        "gemini-1.5-flash": {
            "input": 0.000125,
            "output": 0.000375,
        }
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
- [OpenAI GPT-4V Documentation](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)

### Benchmarks
- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [VQA Challenge](https://visualqa.org/)

### Related Lessons
- [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md)
- [24_API_Evaluation.md](24_API_Evaluation.md)
