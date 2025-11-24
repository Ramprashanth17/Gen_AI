# 🎯 PromptCraft: AI Prompt Analyzer & Optimizer

> A comprehensive system for analyzing, scoring, and optimizing AI prompts using established frameworks.

[![Course](https://img.shields.io/badge/Course-INFO%207390-blue)](https://github.com/nikbearbrown/INFO_7390_Art_and_Science_of_Data)
[![Topic](https://img.shields.io/badge/Topic-Controllable%20Text%20Generation-green)](#)
[![AI Assisted](https://img.shields.io/badge/AI%20Assisted-Claude%20by%20Anthropic-orange)](#)

---

## 📋 Project Overview

As Large Language Models become integral to workflows across industries, the quality of human-AI interaction increasingly depends on **prompt engineering**. PromptCraft addresses this by providing:

1. **Rule-based Analysis**: Scores prompts on 6 universal dimensions
2. **LLM-powered Optimization**: Improves prompts using frameworks like CO-STAR, CRISPE, RISEN, RACE
3. **Interactive Visualization**: Dashboards showing prompt quality breakdown
4. **Educational Content**: Explains *why* certain techniques work

---

## 🎬 Video Demo

[📺 Watch the Demo Video](YOUR_VIDEO_LINK_HERE)

---

## 🗂️ Project Structure

```
promptcraft/
├── notebooks/
│   └── PromptCraft_Crash_Course_GenAI.ipynb    # Main Jupyter notebook
├── webapp/
│   ├── app.py                                   # Streamlit application
│   ├── requirements.txt                         # Python dependencies
│   └── README.md                                # Webapp documentation
├── docs/
│   └── ai_interaction_log.md                    # AI assistance documentation
└── README.md                                    # This file
```

---

## 🚀 Quick Start

### Jupyter Notebook (Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/PromptCraft_Crash_Course_GenAI.ipynb`
3. Add your `GEMINI_API_KEY` to Colab Secrets
4. Run all cells

### Streamlit Web App

```bash
# Install dependencies
cd webapp
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📐 Frameworks Covered

| Framework | Best For | Key Elements |
|-----------|----------|--------------|
| **CO-STAR** | Content creation | Context, Objective, Style, Tone, Audience, Response |
| **CRISPE** | Technical tasks | Capacity, Role, Insight, Statement, Personality, Experiment |
| **RISEN** | Complex tasks | Role, Instructions, Steps, End goal, Narrowing |
| **RACE** | Quick queries | Role, Action, Context, Expectation |

---

## 📊 Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Task Specificity | 25% | Clarity of requested action |
| Context Depth | 20% | Background information provided |
| Role Clarity | 15% | AI persona definition |
| Output Format | 15% | Structure requirements |
| Constraints | 15% | Boundaries and exclusions |
| Style/Tone | 10% | Voice and audience |

---

## 🔬 Technical Approach

### Rule-Based Analysis
- Regex pattern matching for structural elements
- Weighted scoring across dimensions
- No API required

### LLM-Based Evaluation
- Semantic quality assessment via Gemini
- Framework-specific optimization
- Before/after comparison

### Visualization
- Plotly interactive charts
- Gauge meters for overall score
- Radar charts for dimension breakdown

---

## 📚 Learning Outcomes

This project demonstrates understanding of:

1. **Generative AI Fundamentals**
   - Tokenization and context windows
   - Attention mechanisms
   - Model capabilities and limitations

2. **Prompt Engineering**
   - Established frameworks and when to use them
   - Universal dimensions of effective prompts
   - Systematic improvement methodology

3. **Software Development**
   - Python class design
   - API integration
   - Interactive web applications

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations
- **tiktoken** - Tokenization analysis
- **Google Generative AI** - Gemini API
- **Jupyter** - Notebook development

---

## 📖 References

1. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS 2022*.
2. Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP." *ACM Computing Surveys*.
3. Singapore GovTech. "CO-STAR Framework for Prompt Engineering."
4. Anthropic. "Prompt Engineering Guide." https://docs.anthropic.com/claude/docs/prompt-engineering
5. Google. "Prompt Design Strategies." https://ai.google.dev/docs/prompt_best_practices

---

## 📄 License

MIT License

Copyright (c) 2024 Ram Prashanth Rao G

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

---

## 👤 Author

**Ram Prashanth Rao G**
- Course: INFO 7390 - Art and Science of Data
- Term: Fall 2024
- Topic: Controllable Text Generation Techniques

---

## 🙏 Acknowledgments

- Created with AI assistance from Claude (Anthropic)
- Powered by Google's Gemini AI
- Course materials from Professor Nik Bear Brown

---

*Built as part of the "Crash Course in Generative AI" assignment*
