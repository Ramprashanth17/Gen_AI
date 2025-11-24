# PromptCraft: AI Prompt Analyzer & Optimizer

A Streamlit web application that analyzes and optimizes AI prompts using rule-based analysis and LLM-powered optimization.

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# 1. Clone/download the webapp folder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Option 2: Run on Streamlit Cloud (Recommended for Demo)

1. Push the `webapp` folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Option 3: Run on Google Colab

```python
# In a Colab notebook:
!pip install streamlit plotly pandas tiktoken google-generativeai -q

# Write the app.py content to a file
# Then run:
!streamlit run app.py &>/dev/null &
!npx localtunnel --port 8501
```

## 📋 Features

- **Rule-Based Analysis**: Scores prompts on 6 dimensions without API calls
- **LLM Optimization**: Uses Gemini to improve prompts using frameworks
- **Interactive Visualizations**: Gauge charts, radar plots, bar comparisons
- **Multiple Frameworks**: CO-STAR, CRISPE, RISEN, RACE
- **Before/After Comparison**: See exactly how optimization improves your prompt

## 🔑 API Key Setup

To use the optimization features, you need a Gemini API key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new key
4. Enter it in the app's sidebar

## 📁 Project Structure

```
webapp/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 How It Works

### Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Task Specificity | 25% | Clarity of the requested action |
| Context Depth | 20% | Background information provided |
| Role Clarity | 15% | AI persona/expertise definition |
| Output Format | 15% | Structure/length requirements |
| Constraints | 15% | Boundaries and exclusions |
| Style/Tone | 10% | Voice and audience awareness |

### Frameworks

- **CO-STAR**: Best for content creation and communication
- **CRISPE**: Best for technical tasks and code generation
- **RISEN**: Best for complex multi-step tasks
- **RACE**: Best for quick, simple queries

## 📝 License

MIT License - See main project for details.

## 🙏 Acknowledgments

- Built for INFO 7390 - Art and Science of Data
- Created with AI assistance (Claude by Anthropic)
- Powered by Google's Gemini AI
