# 실전 프로젝트(Practical Projects)

> **토픽**: LaTeX
> **레슨**: 16 of 16
> **선수지식**: 모든 이전 레슨 (01-15)
> **목표**: 학습한 모든 개념을 세 가지 완전한 실제 프로젝트에 적용: 학술 논문, Beamer 프레젠테이션, 과학 포스터

## 소개

이 마지막 레슨은 이전 15개 레슨의 모든 내용을 **세 가지 완전하고 컴파일 가능한 프로젝트**로 통합합니다:

1. **학술 논문(Academic Paper)**: 초록, 섹션, 그림, 표, 수식, 참고문헌이 있는 완전한 연구 논문
2. **Beamer 프레젠테이션(Beamer Presentation)**: 사용자 정의 테마, 오버레이, TikZ 다이어그램이 있는 15슬라이드 학회 발표
3. **TikZ 과학 포스터(TikZ Scientific Poster)**: 다중 열 레이아웃, 플롯, QR 코드가 있는 A0 포스터

각 프로젝트는 다음을 포함합니다:
- 완전한 소스 코드
- 컴파일 지침
- 일반적인 함정과 해결책
- 사용자 정의 팁
- 실제 모범 사례

---

## 프로젝트 1: 학술 논문

### 개요

다음에 적합한 완전한 연구 논문 템플릿:
- 학회 제출
- 학술지 기사
- 기술 보고서
- 과정 학기 논문

**특징**:
- 여러 저자와 소속이 있는 제목 페이지
- 초록과 키워드
- 2단 형식
- 하위 섹션이 있는 섹션
- 하위 그림이 있는 그림
- 캡션이 있는 표
- 수학 수식 (번호 있음/없음)
- 알고리즘 의사코드
- 상호 참조
- BibLaTeX를 사용한 참고문헌
- 하이퍼링크

### 완전한 소스 코드

**파일: `paper.tex`**

```latex
\documentclass[conference]{IEEEtran}

% Packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage[backend=biber,style=ieee,sorting=none]{biblatex}
\usepackage[hidelinks]{hyperref}
\usepackage{cleveref}

% Bibliography
\addbibresource{references.bib}

% Custom commands (from Lesson 13)
\newcommand{\R}{\mathbb{R}}
\newcommand{\norm}[1]{\left\| #1 \right\|}
\DeclareMathOperator*{\argmin}{arg\,min}

% Title and authors
\title{Deep Learning for Time Series Forecasting:\\
A Comparative Study of LSTM and Transformer Models}

\author{
  \IEEEauthorblockN{Alice Johnson}
  \IEEEauthorblockA{Department of Computer Science\\
    University of Example\\
    alice.johnson@example.edu}
  \and
  \IEEEauthorblockN{Bob Smith}
  \IEEEauthorblockA{Research Lab XYZ\\
    Institute of Technology\\
    bob.smith@xyz.org}
}

\begin{document}

\maketitle

% Abstract
\begin{abstract}
Time series forecasting is a critical task in many domains including finance, weather prediction, and energy management. Recent advances in deep learning have led to powerful models such as Long Short-Term Memory (LSTM) networks and Transformers. This paper presents a comprehensive comparison of LSTM and Transformer architectures for time series forecasting on three benchmark datasets. We evaluate model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Our experiments show that while Transformers achieve superior accuracy on datasets with long-range dependencies, LSTMs remain competitive on shorter sequences and require significantly less computational resources. We provide implementation details and hyperparameter configurations to facilitate reproducibility.
\end{abstract}

\begin{IEEEkeywords}
Time series forecasting, LSTM, Transformer, deep learning, sequence modeling
\end{IEEEkeywords}

% Introduction
\section{Introduction}
\label{sec:intro}

Time series forecasting aims to predict future values based on historical observations. Traditional methods such as ARIMA \cite{box2015time} and exponential smoothing have been widely used, but recent deep learning approaches have demonstrated superior performance on complex, high-dimensional data.

Long Short-Term Memory (LSTM) networks \cite{hochreiter1997long}, introduced by Hochreiter and Schmidhuber, were specifically designed to capture long-term dependencies in sequential data. More recently, the Transformer architecture \cite{vaswani2017attention}, originally developed for natural language processing, has been adapted for time series tasks \cite{zhou2021informer}.

This paper makes the following contributions:
\begin{itemize}
  \item A systematic comparison of LSTM and Transformer models on three benchmark datasets
  \item Analysis of computational efficiency and scalability
  \item Open-source implementation and trained model weights
  \item Guidelines for practitioners on model selection
\end{itemize}

The rest of this paper is organized as follows. \Cref{sec:related} reviews related work. \Cref{sec:methods} describes the models and datasets. \Cref{sec:results} presents experimental results. \Cref{sec:conclusion} concludes the paper.

% Related Work
\section{Related Work}
\label{sec:related}

\subsection{Classical Methods}

Classical time series forecasting methods include ARIMA \cite{box2015time}, which models temporal dependencies through autoregressive and moving average components. Holt-Winters exponential smoothing extends these ideas to handle trends and seasonality.

\subsection{Deep Learning Approaches}

Recurrent Neural Networks (RNNs) and their variants have become popular for sequence modeling. LSTM networks \cite{hochreiter1997long} address the vanishing gradient problem through gating mechanisms. Gated Recurrent Units (GRUs) \cite{cho2014learning} simplify LSTM architecture while maintaining performance.

\subsection{Attention Mechanisms}

The Transformer architecture \cite{vaswani2017attention} relies entirely on self-attention mechanisms, eliminating recurrence. Informer \cite{zhou2021informer} adapts Transformers for long-sequence time series forecasting with efficient attention mechanisms.

% Methodology
\section{Methodology}
\label{sec:methods}

\subsection{Problem Formulation}

Let $\mathbf{x} = (x_1, x_2, \ldots, x_T) \in \R^T$ denote a univariate time series. The forecasting task is to predict future values $\mathbf{y} = (y_1, y_2, \ldots, y_H)$ given historical observations:
\begin{equation}
  \mathbf{y} = f(\mathbf{x}; \theta)
  \label{eq:forecast}
\end{equation}
where $f$ is the model parameterized by $\theta$, and $H$ is the forecast horizon.

\subsection{LSTM Model}

The LSTM cell computes hidden state $\mathbf{h}_t$ and cell state $\mathbf{c}_t$ as:
\begin{align}
  \mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_f) \label{eq:forget-gate} \\
  \mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_i) \label{eq:input-gate} \\
  \tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_c) \\
  \mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
  \mathbf{o}_t &= \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_o) \\
  \mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{align}
where $\sigma$ is the sigmoid function and $\odot$ denotes element-wise multiplication.

\subsection{Transformer Model}

The Transformer uses multi-head self-attention:
\begin{equation}
  \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
\end{equation}
where $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ are query, key, and value matrices derived from input embeddings.

\subsection{Datasets}

We evaluate models on three benchmark datasets (\cref{tab:datasets}):

\begin{table}[htbp]
\caption{Dataset Statistics}
\label{tab:datasets}
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
Dataset & Samples & Features & Frequency \\
\midrule
ETTh1 & 17,420 & 7 & Hourly \\
Weather & 52,696 & 21 & 10 min \\
Electricity & 26,304 & 321 & Hourly \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Training Procedure}

We minimize the Mean Squared Error (MSE) loss:
\begin{equation}
  L(\theta) = \frac{1}{N} \sum_{i=1}^N \norm{\mathbf{y}^{(i)} - f(\mathbf{x}^{(i)}; \theta)}^2
\end{equation}

Models are trained using Adam optimizer \cite{kingma2014adam} with learning rate $\eta = 10^{-4}$. \Cref{alg:training} summarizes the training procedure.

\begin{algorithm}[htbp]
\caption{Model Training}
\label{alg:training}
\begin{algorithmic}[1]
\Require Dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^N$, epochs $E$
\State Initialize parameters $\theta$
\For{$e = 1$ to $E$}
  \For{each batch $\mathcal{B} \subset \mathcal{D}$}
    \State Compute predictions $\hat{\mathbf{y}} = f(\mathbf{x}; \theta)$ for $\mathbf{x} \in \mathcal{B}$
    \State Compute loss $L = \frac{1}{|\mathcal{B}|} \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{B}} \norm{\mathbf{y} - \hat{\mathbf{y}}}^2$
    \State Update $\theta \leftarrow \theta - \eta \nabla_\theta L$
  \EndFor
\EndFor
\State \Return $\theta$
\end{algorithmic}
\end{algorithm}

% Results
\section{Experimental Results}
\label{sec:results}

\subsection{Quantitative Comparison}

\Cref{fig:results} shows MAE and RMSE for both models across datasets. Transformers achieve lower error on ETTh1 and Weather datasets, while LSTMs perform comparably on Electricity with 50\% fewer parameters.

\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.45\columnwidth}
    \centering
    \includegraphics[width=\textwidth]{mae_comparison.pdf}
    \caption{Mean Absolute Error}
    \label{fig:mae}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\columnwidth}
    \centering
    \includegraphics[width=\textwidth]{rmse_comparison.pdf}
    \caption{Root Mean Squared Error}
    \label{fig:rmse}
  \end{subfigure}
  \caption{Performance comparison on three datasets. Lower is better.}
  \label{fig:results}
\end{figure}

\subsection{Computational Efficiency}

Training time and memory usage are shown in \cref{tab:efficiency}. LSTMs train 2-3× faster than Transformers and use less GPU memory.

\begin{table}[htbp]
\caption{Computational Efficiency (batch size 32)}
\label{tab:efficiency}
\centering
\begin{tabular}{@{}lcc@{}}
\toprule
Model & Training Time (s/epoch) & GPU Memory (GB) \\
\midrule
LSTM & 45.2 & 3.8 \\
Transformer & 128.7 & 7.2 \\
\bottomrule
\end{tabular}
\end{table}

% Conclusion
\section{Conclusion}
\label{sec:conclusion}

This paper presented a comprehensive comparison of LSTM and Transformer models for time series forecasting. Our experiments demonstrate that Transformers excel at capturing long-range dependencies but require more computational resources. LSTMs remain a strong choice for resource-constrained scenarios and shorter sequences.

Future work will explore hybrid architectures combining LSTM and attention mechanisms, and evaluation on multivariate forecasting tasks with more complex dependencies.

% Bibliography
\printbibliography

\end{document}
```

**파일: `references.bib`**

```bibtex
@book{box2015time,
  title={Time series analysis: forecasting and control},
  author={Box, George EP and Jenkins, Gwilym M and Reinsel, Gregory C and Ljung, Greta M},
  year={2015},
  publisher={John Wiley \& Sons}
}

@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  publisher={MIT Press}
}

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@inproceedings{zhou2021informer,
  title={Informer: Beyond efficient transformer for long sequence time-series forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of AAAI},
  year={2021}
}

@article{kingma2014adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  journal={arXiv preprint arXiv:1412.6980},
  year={2014}
}

@article{cho2014learning,
  title={Learning phrase representations using RNN encoder-decoder for statistical machine translation},
  author={Cho, Kyunghyun and Van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1406.1078},
  year={2014}
}
```

### 컴파일

```bash
pdflatex paper.tex
biber paper
pdflatex paper.tex
pdflatex paper.tex
```

또는 `latexmk` 사용:

```bash
latexmk -pdf paper.tex
```

### 일반적인 함정

**문제**: "Undefined references" 또는 PDF에 `??`
- **해결책**: `biber paper` 실행 (not `bibtex`), 그 다음 두 번 더 컴파일

**문제**: 그림이 나타나지 않음
- **해결책**: 플레이스홀더 PDF 생성 또는 `\includegraphics` 줄 주석 처리

**문제**: 2단 수식 오버플로우
- **해결책**: 작은 글꼴로 `equation*` 사용, 또는 `figure*`로 단일 열로 전환

### 사용자 정의 팁

- **단일 열**: `conference` 옵션 제거: `\documentclass{IEEEtran}`
- **다른 참고문헌 스타일**: `style=ieee`를 `style=apa`, `style=nature` 등으로 변경
- **줄 번호 추가**: `\usepackage{lineno}`와 `\begin{document}` 전에 `\linenumbers`
- **블라인드 리뷰**: `\author{}` 주석 처리, `\author{Anonymous}` 사용

---

## 프로젝트 2: Beamer 프레젠테이션

### 개요

다음을 포함하는 15슬라이드 학회 발표:
- 사용자 정의 색상 테마
- 진행 표시기가 있는 섹션 슬라이드
- 오버레이가 있는 콘텐츠 슬라이드 (점진적 공개)
- TikZ 다이어그램
- 코드 목록
- 발표자 노트
- 핸드아웃 생성

### 완전한 소스 코드

**파일: `presentation.tex`**

```latex
\documentclass[aspectratio=169]{beamer}

% Theme
\usetheme{Madrid}
\usecolortheme{default}

% Custom colors
\definecolor{primaryblue}{RGB}{0,82,155}
\definecolor{secondaryorange}{RGB}{255,127,0}
\setbeamercolor{structure}{fg=primaryblue}
\setbeamercolor{alerted text}{fg=secondaryorange}

% Packages
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes,arrows,positioning}
\usepackage{listings}
\usepackage{booktabs}

% Listings style
\lstset{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{primaryblue}\bfseries,
  commentstyle=\color{gray}\itshape,
  stringstyle=\color{secondaryorange},
  showstringspaces=false,
  frame=single
}

% Title
\title{Deep Learning for Time Series Forecasting}
\subtitle{LSTM vs. Transformer: A Comparative Study}
\author{Alice Johnson}
\institute{University of Example}
\date{March 15, 2026}

% Remove navigation symbols
\setbeamertemplate{navigation symbols}{}

% Footer with progress
\setbeamertemplate{footline}[frame number]

\begin{document}

% Title slide
\begin{frame}
  \titlepage
  \note{Welcome everyone. Today I'll present our work on time series forecasting.}
\end{frame}

% Outline
\begin{frame}{Outline}
  \tableofcontents
  \note{Here's what we'll cover in the next 15 minutes.}
\end{frame}

% Section 1
\section{Introduction}

\begin{frame}{Motivation}
  \begin{block}{Time Series Forecasting}
    Predicting future values based on historical observations
  \end{block}

  \vspace{0.5cm}

  \textbf{Applications}:
  \begin{itemize}
    \item<2-> Finance: Stock price prediction
    \item<3-> Energy: Electricity demand forecasting
    \item<4-> Weather: Temperature and precipitation
    \item<5-> Healthcare: Patient monitoring and early warning
  \end{itemize}

  \note<1->{Introduce the problem}
  \note<2->{Financial applications are critical}
  \note<3->{Energy sector needs accurate forecasts for grid management}
\end{frame}

\begin{frame}{Research Questions}
  \begin{enumerate}
    \item How do LSTM and Transformer models compare in accuracy?
    \item What are the computational trade-offs?
    \item Which model should practitioners choose?
  \end{enumerate}

  \vspace{1cm}

  \pause

  \begin{alertblock}{Hypothesis}
    Transformers achieve better accuracy on long sequences, but LSTMs are more efficient.
  \end{alertblock}
\end{frame}

% Section 2
\section{Methodology}

\begin{frame}{Model Architectures}
  \begin{columns}[T]
    \begin{column}{0.48\textwidth}
      \centering
      \textbf{LSTM}
      \vspace{0.3cm}

      \begin{tikzpicture}[scale=0.7,
        node/.style={rectangle,draw,minimum width=1.5cm,minimum height=0.8cm}]
        \node[node] (x) at (0,0) {Input};
        \node[node] (lstm1) at (0,1.5) {LSTM};
        \node[node] (lstm2) at (0,3) {LSTM};
        \node[node] (fc) at (0,4.5) {Dense};
        \node[node] (y) at (0,6) {Output};

        \draw[->] (x) -- (lstm1);
        \draw[->] (lstm1) -- (lstm2);
        \draw[->] (lstm2) -- (fc);
        \draw[->] (fc) -- (y);
      \end{tikzpicture}
    \end{column}

    \begin{column}{0.48\textwidth}
      \centering
      \textbf{Transformer}
      \vspace{0.3cm}

      \begin{tikzpicture}[scale=0.7,
        node/.style={rectangle,draw,minimum width=1.5cm,minimum height=0.8cm}]
        \node[node] (x) at (0,0) {Input};
        \node[node] (emb) at (0,1.5) {Embedding};
        \node[node] (att1) at (0,3) {Attention};
        \node[node] (att2) at (0,4.5) {Attention};
        \node[node] (y) at (0,6) {Output};

        \draw[->] (x) -- (emb);
        \draw[->] (emb) -- (att1);
        \draw[->] (att1) -- (att2);
        \draw[->] (att2) -- (y);
      \end{tikzpicture}
    \end{column}
  \end{columns}

  \note{LSTM processes sequences recurrently; Transformer uses parallel attention.}
\end{frame}

\begin{frame}[fragile]{LSTM Equations}
  The LSTM cell updates hidden state via gating:

  \begin{align*}
    \mathbf{f}_t &= \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_f) \\
    \mathbf{i}_t &= \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, x_t] + \mathbf{b}_i) \\
    \mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
  \end{align*}

  \vspace{0.5cm}

  \pause

  \begin{lstlisting}[language=Python]
import torch.nn as nn

lstm = nn.LSTM(input_size=10, hidden_size=128,
               num_layers=2, batch_first=True)
  \end{lstlisting}
\end{frame}

\begin{frame}{Datasets}
  \begin{table}
    \centering
    \begin{tabular}{@{}lccc@{}}
      \toprule
      Dataset & Samples & Features & Frequency \\
      \midrule
      ETTh1 & 17,420 & 7 & Hourly \\
      Weather & 52,696 & 21 & 10 min \\
      Electricity & 26,304 & 321 & Hourly \\
      \bottomrule
    \end{tabular}
  \end{table}

  \vspace{0.5cm}

  \begin{itemize}
    \item Split: 70\% train, 15\% validation, 15\% test
    \item Metrics: MAE, RMSE
  \end{itemize}
\end{frame}

% Section 3
\section{Results}

\begin{frame}{Accuracy Comparison}
  \centering
  \includegraphics[width=0.7\textwidth]{mae_comparison.pdf}

  \vspace{0.3cm}

  \begin{itemize}
    \item Transformer: 12\% lower MAE on ETTh1
    \item LSTM: Competitive on Electricity dataset
  \end{itemize}

  \note{Highlight the accuracy advantage of Transformers on long sequences.}
\end{frame}

\begin{frame}{Computational Efficiency}
  \begin{columns}[T]
    \begin{column}{0.5\textwidth}
      \textbf{Training Time}
      \begin{itemize}
        \item LSTM: 45 s/epoch
        \item Transformer: 129 s/epoch
        \item \alert{2.9× faster}
      \end{itemize}
    \end{column}

    \begin{column}{0.5\textwidth}
      \textbf{GPU Memory}
      \begin{itemize}
        \item LSTM: 3.8 GB
        \item Transformer: 7.2 GB
        \item \alert{1.9× less memory}
      \end{itemize}
    \end{column}
  \end{columns}

  \vspace{1cm}

  \begin{block}{Key Insight}
    LSTMs offer significant computational savings with minor accuracy trade-offs.
  \end{block}
\end{frame}

% Section 4
\section{Conclusion}

\begin{frame}{Summary}
  \textbf{Contributions}:
  \begin{itemize}
    \item Systematic comparison on 3 benchmark datasets
    \item Analysis of accuracy vs. efficiency trade-offs
    \item Open-source code and trained models
  \end{itemize}

  \vspace{0.8cm}

  \pause

  \textbf{Recommendations}:
  \begin{itemize}
    \item Use \alert{Transformer} for maximum accuracy on long sequences
    \item Use \alert{LSTM} for resource-constrained environments
    \item Consider hybrid models for best of both worlds
  \end{itemize}
\end{frame}

\begin{frame}{Future Work}
  \begin{enumerate}
    \item Hybrid LSTM-Transformer architectures
    \item Multivariate forecasting with graph neural networks
    \item Real-time inference optimization
    \item Application to finance and healthcare domains
  \end{enumerate}

  \vspace{1cm}

  \centering
  \Large \textbf{Thank you!}

  \vspace{0.5cm}

  \normalsize
  Questions? \\
  \texttt{alice.johnson@example.edu}
\end{frame}

% Backup slides
\appendix

\begin{frame}{Backup: Hyperparameters}
  \begin{table}
    \centering
    \small
    \begin{tabular}{@{}lcc@{}}
      \toprule
      Parameter & LSTM & Transformer \\
      \midrule
      Hidden size & 128 & 256 \\
      Layers & 2 & 4 \\
      Dropout & 0.2 & 0.1 \\
      Learning rate & $10^{-4}$ & $10^{-4}$ \\
      Batch size & 32 & 32 \\
      \bottomrule
    \end{tabular}
  \end{table}
\end{frame}

\end{document}
```

### 컴파일

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

### 핸드아웃 생성

이 옵션 추가:

```latex
\documentclass[aspectratio=169,handout]{beamer}
```

평소대로 컴파일. 오버레이가 축소됩니다.

### 발표자 노트

주석을 지원하는 PDF 뷰어에서 노트 보기, 또는 다음 사용:

```bash
pdfpc presentation.pdf
```

(`pdfpc` 도구 필요)

### 일반적인 함정

**문제**: 오버레이가 작동하지 않음
- **해결책**: `\pause`, `\only<2->`, `\item<3->` 구문을 올바르게 사용

**문제**: 슬라이드당 텍스트가 너무 많음
- **해결책**: "6×6 규칙" 따르기: 최대 6개 글머리 기호, 각 6단어

**문제**: TikZ 다이어그램이 너무 복잡함
- **해결책**: 단순화하거나 외부 도구에서 생성, PDF로 가져오기

---

## 프로젝트 3: TikZ 과학 포스터

### 개요

다음을 포함하는 학회용 A0 포스터 (841 × 1189 mm):
- 다중 열 레이아웃 (3열)
- 로고가 있는 제목 배너
- 서론, 방법, 결과, 결론 블록
- 데이터 시각화를 위한 PGFPlots
- TikZ 순서도
- 참조용 QR 코드
- 사용자 정의 색상 체계

### 완전한 소스 코드

**파일: `poster.tex`**

```latex
\documentclass[a0paper,portrait]{tikzposter}

% Packages
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{qrcode}

% Theme
\usetheme{Default}
\usecolorstyle{Denmark}

% Custom colors
\definecolor{primaryblue}{RGB}{0,82,155}
\definecolor{lightblue}{RGB}{204,229,255}
\definecolor{darkgray}{RGB}{51,51,51}

\colorlet{backgroundcolor}{lightblue!30}
\colorlet{blocktitlefgcolor}{white}
\colorlet{blocktitlebgcolor}{primaryblue}
\colorlet{blockbodyfgcolor}{darkgray}
\colorlet{blockbodybgcolor}{white}

% Title
\title{\parbox{0.8\linewidth}{\centering Deep Learning for Time Series Forecasting: LSTM vs. Transformer}}
\author{Alice Johnson$^1$, Bob Smith$^2$}
\institute{$^1$University of Example, $^2$Research Lab XYZ}

% Logos (placeholders - replace with actual logos)
\titlegraphic{
  \includegraphics[width=0.1\textwidth]{logo1.pdf}
  \hspace{2cm}
  \includegraphics[width=0.1\textwidth]{logo2.pdf}
}

\begin{document}

\maketitle

\begin{columns}
  % Column 1
  \column{0.33}

  \block{Introduction}{
    \textbf{Motivation:} Time series forecasting is critical in finance, energy, and healthcare.

    \vspace{0.5cm}

    \textbf{Problem:} Traditional methods (ARIMA, exponential smoothing) struggle with complex, high-dimensional data.

    \vspace{0.5cm}

    \textbf{Solution:} Deep learning models (LSTM, Transformer) capture nonlinear dependencies.

    \vspace{0.5cm}

    \textbf{Research Questions:}
    \begin{itemize}
      \item How do LSTM and Transformer compare in accuracy?
      \item What are computational trade-offs?
      \item Which model should practitioners choose?
    \end{itemize}
  }

  \block{Model Architectures}{
    \textbf{LSTM:} Recurrent architecture with gating mechanisms
    \[
      \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
    \]

    \vspace{1cm}

    \textbf{Transformer:} Self-attention for parallel sequence processing
    \[
      \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
    \]

    \vspace{1cm}

    \begin{tikzfigure}[Comparison of architectures]
      \begin{tikzpicture}[scale=1.5,
        node/.style={rectangle,draw,minimum width=2.5cm,minimum height=1cm,fill=white}]

        % LSTM
        \node[node] (x1) at (0,0) {Input};
        \node[node] (lstm1) at (0,2) {LSTM Layer 1};
        \node[node] (lstm2) at (0,4) {LSTM Layer 2};
        \node[node] (fc1) at (0,6) {Dense};
        \node[node] (y1) at (0,8) {Output};

        \draw[->,thick] (x1) -- (lstm1);
        \draw[->,thick] (lstm1) -- (lstm2);
        \draw[->,thick] (lstm2) -- (fc1);
        \draw[->,thick] (fc1) -- (y1);

        \node at (0,-1) {\textbf{LSTM}};

        % Transformer
        \node[node] (x2) at (6,0) {Input};
        \node[node] (emb) at (6,2) {Embedding};
        \node[node] (att1) at (6,4) {Attention $\times$ 4};
        \node[node] (fc2) at (6,6) {Dense};
        \node[node] (y2) at (6,8) {Output};

        \draw[->,thick] (x2) -- (emb);
        \draw[->,thick] (emb) -- (att1);
        \draw[->,thick] (att1) -- (fc2);
        \draw[->,thick] (fc2) -- (y2);

        \node at (6,-1) {\textbf{Transformer}};
      \end{tikzpicture}
    \end{tikzfigure}
  }

  % Column 2
  \column{0.33}

  \block{Datasets}{
    \begin{tabular}{@{}lccc@{}}
      \toprule
      Dataset & Samples & Features & Frequency \\
      \midrule
      ETTh1 & 17,420 & 7 & Hourly \\
      Weather & 52,696 & 21 & 10 min \\
      Electricity & 26,304 & 321 & Hourly \\
      \bottomrule
    \end{tabular}

    \vspace{0.5cm}

    \textbf{Split:} 70\% train, 15\% validation, 15\% test

    \textbf{Metrics:} Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
  }

  \block{Results: Accuracy}{
    \begin{tikzfigure}[MAE comparison across datasets]
      \begin{tikzpicture}
        \begin{axis}[
          ybar,
          width=0.9\linewidth,
          height=10cm,
          ylabel={Mean Absolute Error (MAE)},
          xlabel={Dataset},
          symbolic x coords={ETTh1, Weather, Electricity},
          xtick=data,
          legend pos=north west,
          ymajorgrids=true,
          bar width=0.8cm,
          enlarge x limits=0.2,
          ymin=0
        ]
        \addplot coordinates {(ETTh1,0.42) (Weather,0.31) (Electricity,0.18)};
        \addplot coordinates {(ETTh1,0.37) (Weather,0.28) (Electricity,0.19)};
        \legend{LSTM, Transformer}
        \end{axis}
      \end{tikzpicture}
    \end{tikzfigure}

    \textbf{Key Finding:} Transformer achieves 12\% lower MAE on ETTh1 (long sequences)
  }

  % Column 3
  \column{0.33}

  \block{Results: Efficiency}{
    \begin{tikzfigure}[Training time and memory usage]
      \begin{tikzpicture}
        \begin{axis}[
          ybar,
          width=0.9\linewidth,
          height=8cm,
          ylabel={Training Time (s/epoch)},
          symbolic x coords={LSTM, Transformer},
          xtick=data,
          bar width=1.5cm,
          ymin=0,
          ymajorgrids=true,
          nodes near coords,
          enlarge x limits=0.5
        ]
        \addplot coordinates {(LSTM,45.2) (Transformer,128.7)};
        \end{axis}
      \end{tikzpicture}
    \end{tikzfigure}

    \vspace{1cm}

    \textbf{Computational Trade-offs:}
    \begin{itemize}
      \item LSTM: 2.9$\times$ faster training
      \item LSTM: 1.9$\times$ less GPU memory
      \item Transformer: Better accuracy on long sequences
    \end{itemize}
  }

  \block{Conclusion}{
    \textbf{Contributions:}
    \begin{itemize}
      \item Comprehensive comparison on 3 datasets
      \item Accuracy vs. efficiency trade-off analysis
      \item Open-source implementation
    \end{itemize}

    \vspace{0.5cm}

    \textbf{Recommendations:}
    \begin{itemize}
      \item \textbf{Transformer:} Maximum accuracy, long sequences
      \item \textbf{LSTM:} Resource-constrained environments
    \end{itemize}

    \vspace{1cm}

    \textbf{Future Work:} Hybrid architectures, multivariate forecasting, real-time inference

    \vspace{1cm}

    \begin{center}
      \textbf{Code \& Data:}\\
      \qrcode[height=3cm]{https://github.com/example/time-series-forecast}
    \end{center}

    \vspace{0.5cm}

    \begin{center}
      \textbf{Contact:} \texttt{alice.johnson@example.edu}
    \end{center}
  }

\end{columns}

\end{document}
```

### 컴파일

```bash
pdflatex poster.tex
```

**참고**: TikZ 복잡도로 인해 컴파일 시간이 더 걸릴 수 있습니다.

### 인쇄

실제 학회 포스터의 경우:
1. PDF로 내보내기
2. 전문 포스터 인쇄 서비스로 전송
3. 지정: A0 크기, 세로, 고품질 (600 dpi)

### 일반적인 함정

**문제**: 인쇄 시 텍스트가 너무 작음
- **해결책**: `\tikzposter` 옵션에서 더 큰 글꼴 크기 사용

**문제**: QR 코드가 스캔되지 않음
- **해결책**: `height` 매개변수 증가, 인쇄 전 테스트

**문제**: 화면과 인쇄물의 색상이 다름
- **해결책**: CMYK 색상 공간 사용, 인쇄 시뮬레이션으로 미리보기

---

## 레슨 통합: 통합 맵

세 가지 프로젝트 모두 이전 레슨의 개념을 사용합니다:

| 레슨 | 프로젝트 1 (논문) | 프로젝트 2 (Beamer) | 프로젝트 3 (포스터) |
|--------|-------------------|--------------------|--------------------|
| L01-02 | 문서 구조 | 프레임 구조 | 블록 구조 |
| L03 | 텍스트 서식 | 테마 색상 | 사용자 정의 색상 |
| L05 | 표 (booktabs) | 표 | 표 |
| L06 | 그림, 하위 그림 | 이미지 | TikZ 그림 |
| L07-08 | 수식, align | 슬라이드의 수학 | 블록의 수학 |
| L09 | 상호 참조 | 프레임 참조 | — |
| L10 | BibLaTeX | 인용 | — |
| L11 | — | Beamer 테마, 오버레이 | — |
| L12 | — | TikZ 다이어그램 | PGFPlots, TikZ |
| L13 | 사용자 정의 명령 | — | — |
| L14 | IEEE 클래스 | Beamer 클래스 | tikzposter 클래스 |
| L15 | latexmk | latexmk | — |

---

## 다음 단계

### 고급 주제 탐색

1. **LuaLaTeX 프로그래밍**: 복잡한 문서 생성 자동화
2. **외부화(Externalization)**: TikZ 컴파일 속도 향상
3. **ConTeXt**: 고급 타이포그래피를 위한 LaTeX 대안
4. **arXiv 제출**: 사전 인쇄 서버용 논문 준비
5. **저널별 템플릿**: IEEE, ACM, Springer, Elsevier

### 커뮤니티 참여

- **TeX StackExchange**: 문제 해결을 위한 Q&A
- **LaTeX Project**: 공식 뉴스 및 릴리스
- **CTAN**: 6000개 이상의 패키지 탐색
- **Overleaf 튜토리얼**: 비디오 가이드 및 웨비나
- **로컬 TeX 사용자 그룹**: TUG, UK-TUG 등

### 연습 프로젝트

- LaTeX로 이력서 작성
- 다가오는 발표를 위한 프레젠테이션 생성
- 노트나 문서 조판
- 오픈 소스 LaTeX 패키지에 기여

---

## 요약

이 레슨은 세 가지 완전한 실제 LaTeX 프로젝트를 제시했습니다:

1. **학술 논문(Academic Paper)**: 그림, 표, 수학, 참고문헌이 있는 IEEE 스타일 학회 논문
2. **Beamer 프레젠테이션(Beamer Presentation)**: 오버레이, TikZ, 사용자 정의 테마가 있는 15슬라이드 발표
3. **과학 포스터(Scientific Poster)**: 다중 열 레이아웃, 플롯, QR 코드가 있는 A0 포스터

**시연된 핵심 기술**:
- 문서 클래스 선택 및 구성
- 패키지 통합 (그래픽, 수학, 참고문헌, TikZ)
- 사용자 정의 명령 및 환경
- 상호 참조 및 인용
- 시각적 디자인 (색상, 레이아웃, 테마)
- 컴파일 워크플로우

**축하합니다!** LaTeX 과정의 16개 레슨을 모두 완료했습니다. 이제 학술 및 전문 맥락에서 전문 문서, 프레젠테이션, 포스터를 생성하는 기술을 갖추었습니다.

---

**탐색**

- 이전: [15_Automation_and_Build.md](15_Automation_and_Build.md)
- 과정 종료
