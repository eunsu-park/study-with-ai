# MCP Server Recommendations / MCP 서버 추천

Recommended MCP (Model Context Protocol) servers to enhance the Study with AI workflow.
Study with AI 워크플로우를 향상시키기 위한 추천 MCP 서버 목록입니다.

**Last updated / 최종 수정**: 2026-04-02

---

## 1. Paper Search MCP / 논문 검색 MCP

**Package**: `paper-search-mcp`

**Purpose / 목적**:
Search and download academic papers from 20+ sources including arXiv, Semantic Scholar, CrossRef, Unpaywall, PubMed, and more.
arXiv, Semantic Scholar, CrossRef, Unpaywall, PubMed 등 20개 이상의 소스에서 학술 논문을 검색하고 다운로드합니다.

**How it improves the workflow / 워크플로우 개선 방법**:
Replaces the manual PDF download process (Step 1 in the workflow). Claude can search by title, DOI, or keywords and download PDFs automatically.
워크플로우의 Step 1 (PDF 다운로드)을 대체합니다. Claude가 제목, DOI, 키워드로 검색하고 자동으로 PDF를 다운로드할 수 있습니다.

**Installation / 설치**:
```bash
pip install paper-search-mcp
claude mcp add paper-search --transport stdio -- paper-search-mcp
```

**Key tools / 주요 도구**:
- `search_papers` — Search across multiple academic databases / 여러 학술 데이터베이스에서 검색
- `get_paper_details` — Get metadata (authors, abstract, DOI) / 메타데이터 조회
- `download_paper` — Download PDF to local path / PDF를 로컬 경로에 다운로드

---

## 2. PDF Reader MCP / PDF 리더 MCP

**Package**: `@fabriqa.ai/pdf-reader-mcp`

**Purpose / 목적**:
Extract text from PDF files page-by-page without loading the entire document into context. Supports search and metadata extraction.
전체 문서를 context에 로드하지 않고 PDF 파일에서 페이지별로 텍스트를 추출합니다. 검색 및 메타데이터 추출을 지원합니다.

**How it improves the workflow / 워크플로우 개선 방법**:
Replaces the screenshot upload workflow (Step 3). Claude can read paper PDFs directly, extract specific pages, and search for terms within the paper.
스크린샷 업로드 워크플로우 (Step 3)를 대체합니다. Claude가 논문 PDF를 직접 읽고, 특정 페이지를 추출하며, 논문 내 용어를 검색할 수 있습니다.

**Installation / 설치**:
```bash
claude mcp add pdf-reader --transport stdio -- npx -y @fabriqa.ai/pdf-reader-mcp
```

**Key tools / 주요 도구**:
- `read_pdf` — Extract text from specific pages / 특정 페이지에서 텍스트 추출
- `search_pdf` — Search for terms within a PDF / PDF 내 용어 검색
- `get_pdf_metadata` — Get document metadata / 문서 메타데이터 조회

---

## 3. Jupyter Notebook MCP / Jupyter 노트북 MCP

**Package**: `jupyter-notebook-mcp`

**Purpose / 목적**:
Create, modify, and execute Jupyter notebooks programmatically. Allows Claude to run code cells and verify outputs.
프로그래밍 방식으로 Jupyter 노트북을 생성, 수정, 실행합니다. Claude가 코드 셀을 실행하고 출력을 검증할 수 있습니다.

**How it improves the workflow / 워크플로우 개선 방법**:
Enhances Step 5 (implementation.ipynb creation). Claude can execute code to verify implementations produce correct results before saving.
Step 5 (implementation.ipynb 생성)를 향상시킵니다. Claude가 코드를 실행하여 구현이 올바른 결과를 생성하는지 저장 전에 검증할 수 있습니다.

**Installation / 설치**:
```bash
git clone https://github.com/jjsantos01/jupyter-notebook-mcp
cd jupyter-notebook-mcp
pip install -e .
claude mcp add jupyter --transport stdio -- jupyter-notebook-mcp
```

**Key tools / 주요 도구**:
- `create_notebook` — Create a new notebook / 새 노트북 생성
- `add_cell` — Add markdown or code cells / 마크다운 또는 코드 셀 추가
- `execute_cell` — Run a code cell and return output / 코드 셀 실행 및 출력 반환

---

## Quick Setup (All Three) / 빠른 설치 (전체)

```bash
# 1. Paper Search / 논문 검색
pip install paper-search-mcp
claude mcp add paper-search --transport stdio -- paper-search-mcp

# 2. PDF Reader / PDF 리더
claude mcp add pdf-reader --transport stdio -- npx -y @fabriqa.ai/pdf-reader-mcp

# 3. Jupyter Notebook / Jupyter 노트북
pip install jupyter-notebook-mcp
claude mcp add jupyter --transport stdio -- jupyter-notebook-mcp
```

## Verification / 설치 확인

After installation, verify MCP servers are available:
설치 후 MCP 서버 사용 가능 여부를 확인합니다:

```bash
claude mcp list
```

All three servers should appear in the list.
세 서버 모두 목록에 나타나야 합니다.
