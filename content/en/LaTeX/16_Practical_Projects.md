# Practical Projects

> **Topic**: LaTeX
> **Lesson**: 16 of 16
> **Prerequisites**: All previous lessons (01-15)
> **Objective**: Apply all learned concepts to three complete, real-world projects: an academic paper, a Beamer presentation, and a scientific poster

## Introduction

This final lesson brings together everything from the previous 15 lessons into **three complete, compilable projects**:

1. **Academic Paper**: Full research paper with abstract, sections, figures, tables, equations, and bibliography
2. **Beamer Presentation**: 15-slide conference presentation with custom theme, overlays, and TikZ diagrams
3. **TikZ Scientific Poster**: A0 poster with multi-column layout, plots, and QR code

Each project includes:
- Complete source code
- Compilation instructions
- Common pitfalls and solutions
- Customization tips
- Real-world best practices

---

## Project 1: Academic Paper

### Overview

A complete research paper template suitable for:
- Conference submissions
- Journal articles
- Technical reports
- Course term papers

**Features**:
- Title page with multiple authors and affiliations
- Abstract and keywords
- Two-column format
- Sections with subsections
- Figures with subfigures
- Tables with captions
- Mathematical equations (numbered and unnumbered)
- Algorithm pseudocode
- Cross-references
- Bibliography with BibLaTeX
- Hyperlinks

### Complete Source Code

**File: `paper.tex`**

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

**File: `references.bib`**

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

### Compilation

```bash
pdflatex paper.tex
biber paper
pdflatex paper.tex
pdflatex paper.tex
```

Or with `latexmk`:

```bash
latexmk -pdf paper.tex
```

### Common Pitfalls

**Problem**: "Undefined references" or `??` in PDF
- **Solution**: Run `biber paper` (not `bibtex`), then compile twice more

**Problem**: Figures don't appear
- **Solution**: Create placeholder PDFs or comment out `\includegraphics` lines

**Problem**: Two-column equations overflow
- **Solution**: Use `equation*` with smaller font, or switch to one-column with `figure*`

### Customization Tips

- **Single column**: Remove `conference` option: `\documentclass{IEEEtran}`
- **Different bibliography style**: Change `style=ieee` to `style=apa`, `style=nature`, etc.
- **Add line numbers**: `\usepackage{lineno}` and `\linenumbers` before `\begin{document}`
- **Blind review**: Comment out `\author{}`, use `\author{Anonymous}`

---

## Project 2: Beamer Presentation

### Overview

A 15-slide conference presentation with:
- Custom color theme
- Section slides with progress indicators
- Content slides with overlays (incremental reveals)
- TikZ diagrams
- Code listings
- Speaker notes
- Handout generation

### Complete Source Code

**File: `presentation.tex`**

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

### Compilation

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

### Creating Handouts

Add this option:

```latex
\documentclass[aspectratio=169,handout]{beamer}
```

Compile as usual. Overlays will be collapsed.

### Speaker Notes

View notes in PDF viewer supporting annotations, or use:

```bash
pdfpc presentation.pdf
```

(Requires `pdfpc` tool)

### Common Pitfalls

**Problem**: Overlays don't work
- **Solution**: Use `\pause`, `\only<2->`, `\item<3->` syntax correctly

**Problem**: Too much text per slide
- **Solution**: Follow "6×6 rule": max 6 bullets, 6 words each

**Problem**: TikZ diagrams too complex
- **Solution**: Simplify or create in external tool, import as PDF

---

## Project 3: TikZ Scientific Poster

### Overview

An A0 poster (841 × 1189 mm) for a conference, featuring:
- Multi-column layout (3 columns)
- Title banner with logos
- Introduction, methods, results, conclusion blocks
- PGFPlots for data visualization
- TikZ flowchart
- QR code for references
- Custom color scheme

### Complete Source Code

**File: `poster.tex`**

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

### Compilation

```bash
pdflatex poster.tex
```

**Note**: Compile may take longer due to TikZ complexity.

### Printing

For actual conference poster:
1. Export to PDF
2. Send to professional poster printing service
3. Specify: A0 size, portrait, high-quality (600 dpi)

### Common Pitfalls

**Problem**: Text too small when printed
- **Solution**: Use larger font sizes in `\tikzposter` options

**Problem**: QR code doesn't scan
- **Solution**: Increase `height` parameter, test before printing

**Problem**: Colors look different on screen vs. print
- **Solution**: Use CMYK color space, preview with print simulation

---

## Combining Lessons: Integration Map

All three projects use concepts from previous lessons:

| Lesson | Project 1 (Paper) | Project 2 (Beamer) | Project 3 (Poster) |
|--------|-------------------|--------------------|--------------------|
| L01-02 | Document structure | Frame structure | Block structure |
| L03 | Text formatting | Theme colors | Custom colors |
| L05 | Tables (booktabs) | Tables | Tables |
| L06 | Figures, subfigures | Images | TikZ figures |
| L07-08 | Equations, align | Math in slides | Math in blocks |
| L09 | Cross-references | Frame references | — |
| L10 | BibLaTeX | Citations | — |
| L11 | — | Beamer themes, overlays | — |
| L12 | — | TikZ diagrams | PGFPlots, TikZ |
| L13 | Custom commands | — | — |
| L14 | IEEE class | Beamer class | tikzposter class |
| L15 | latexmk | latexmk | — |

---

## Next Steps

### Explore Advanced Topics

1. **LuaLaTeX programming**: Automate complex document generation
2. **Externalization**: Speed up TikZ compilation
3. **ConTeXt**: Alternative to LaTeX for advanced typography
4. **arXiv submission**: Prepare papers for preprint servers
5. **Journal-specific templates**: IEEE, ACM, Springer, Elsevier

### Join the Community

- **TeX StackExchange**: Q&A for troubleshooting
- **LaTeX Project**: Official news and releases
- **CTAN**: Explore 6000+ packages
- **Overleaf tutorials**: Video guides and webinars
- **Local TeX user groups**: TUG, UK-TUG, etc.

### Practice Projects

- Write your CV in LaTeX
- Create presentation for upcoming talk
- Typeset notes or documentation
- Contribute to open-source LaTeX packages

---

## Summary

This lesson presented three complete, real-world LaTeX projects:

1. **Academic Paper**: IEEE-style conference paper with figures, tables, math, bibliography
2. **Beamer Presentation**: 15-slide talk with overlays, TikZ, custom theme
3. **Scientific Poster**: A0 poster with multi-column layout, plots, QR code

**Key skills demonstrated**:
- Document class selection and configuration
- Package integration (graphics, math, bibliography, TikZ)
- Custom commands and environments
- Cross-referencing and citations
- Visual design (colors, layouts, themes)
- Compilation workflows

**Congratulations!** You've completed all 16 lessons of the LaTeX course. You now have the skills to create professional documents, presentations, and posters for academic and professional contexts.

---

**Navigation**

- Previous: [15_Automation_and_Build.md](15_Automation_and_Build.md)
- End of Course
