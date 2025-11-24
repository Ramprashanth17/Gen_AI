"""
PromptCraft: AI Prompt Analyzer & Optimizer
============================================
A Streamlit web application for analyzing and improving AI prompts.

This app demonstrates:
- Rule-based prompt analysis
- LLM-powered optimization (Gemini)
- Interactive visualization of prompt quality
- Framework-based prompt engineering

Author: [Your Name]
Course: INFO 7390 - Art and Science of Data
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import tiktoken

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PromptCraft - AI Prompt Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS FOR STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .score-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .grade-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .grade-a { background: #10B981; color: white; }
    .grade-b { background: #3B82F6; color: white; }
    .grade-c { background: #F59E0B; color: white; }
    .grade-d { background: #EF4444; color: white; }
    .grade-f { background: #6B7280; color: white; }
    
    /* Dimension cards */
    .dimension-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Framework selector */
    .framework-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .framework-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    /* Optimized prompt box */
    .optimized-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #22c55e;
        margin-top: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    score: float
    max_score: float
    feedback: str
    suggestions: List[str]
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0

# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT ANALYZER (Rule-Based)
# ═══════════════════════════════════════════════════════════════════════════════

class PromptAnalyzer:
    """
    Rule-based prompt analyzer that scores prompts on six dimensions.
    """
    
    WEIGHTS = {
        'task_specificity': 25,
        'context_depth': 20,
        'role_clarity': 15,
        'output_format': 15,
        'constraints': 15,
        'style_tone': 10
    }
    
    ROLE_PATTERNS = [
        r'\b(you are|act as|assume the role|as a|pretend to be|imagine you\'re)\b',
        r'\b(expert|specialist|professional|consultant|advisor|coach|mentor)\b',
        r'\brole\s*:',
        r'\b(developer|engineer|analyst|scientist|writer|designer)\b'
    ]
    
    CONTEXT_PATTERNS = [
        r'\b(context|background|situation|scenario)\s*:',
        r'\b(i\'m working on|i need|my goal|i have|currently)\b',
        r'\b(because|since|given that|considering)\b',
        r'\b(project|task|assignment|problem)\b'
    ]
    
    TASK_PATTERNS = [
        r'\b(task|objective|goal|please|can you|could you|i need you to)\b',
        r'\b(write|create|generate|explain|analyze|summarize|list|describe)\b',
        r'\b(help me|assist|provide|give me)\b',
        r'\b(step[s]?|instruction[s]?)\b'
    ]
    
    FORMAT_PATTERNS = [
        r'\b(format|structure|output)\s*:',
        r'\b(bullet|numbered|list|table|json|markdown|code)\b',
        r'\b(\d+\s*words?|\d+\s*sentences?|\d+\s*paragraphs?)\b',
        r'\b(sections?|headers?|organize)\b'
    ]
    
    CONSTRAINT_PATTERNS = [
        r'\b(do not|don\'t|avoid|exclude|never|without)\b',
        r'\b(constraint|limit|restriction|boundary)\b',
        r'\b(only|must|should not|refrain)\b',
        r'\b(maximum|minimum|at most|at least)\b'
    ]
    
    STYLE_PATTERNS = [
        r'\b(tone|style|voice|manner)\s*:',
        r'\b(formal|informal|casual|professional|friendly|serious)\b',
        r'\b(simple|technical|beginner|advanced|detailed|brief)\b',
        r'\b(audience|readers?|users?)\b'
    ]
    
    def __init__(self):
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def analyze(self, prompt: str) -> Dict:
        """Perform comprehensive analysis of a prompt."""
        prompt_lower = prompt.lower()
        
        # Token count
        token_count = len(self.encoding.encode(prompt)) if self.encoding else len(prompt.split())
        
        # Score each dimension
        dimensions = {
            'task_specificity': self._score_task(prompt, prompt_lower),
            'context_depth': self._score_context(prompt, prompt_lower),
            'role_clarity': self._score_role(prompt, prompt_lower),
            'output_format': self._score_format(prompt, prompt_lower),
            'constraints': self._score_constraints(prompt, prompt_lower),
            'style_tone': self._score_style(prompt, prompt_lower)
        }
        
        total_score = sum(d.score for d in dimensions.values())
        
        return {
            'total_score': round(total_score, 1),
            'grade': self._score_to_grade(total_score),
            'token_count': token_count,
            'dimensions': dimensions,
            'feedback': self._generate_feedback(total_score)
        }
    
    def _count_patterns(self, text: str, patterns: List[str]) -> int:
        return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    
    def _score_role(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['role_clarity']
        matches = self._count_patterns(prompt_lower, self.ROLE_PATTERNS)
        
        if matches >= 3:
            score, feedback = max_score, "Excellent role definition"
            suggestions = []
        elif matches >= 2:
            score, feedback = max_score * 0.75, "Good role indication"
            suggestions = ["Specify expertise level"]
        elif matches >= 1:
            score, feedback = max_score * 0.4, "Basic role mentioned"
            suggestions = ["Add 'You are a [role]' at start", "Specify relevant expertise"]
        else:
            score, feedback = max_score * 0.1, "No role defined"
            suggestions = ["Start with 'You are a [role]'", "Define AI's expertise"]
        
        return DimensionScore("Role Clarity", round(score, 1), max_score, feedback, suggestions)
    
    def _score_context(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['context_depth']
        matches = self._count_patterns(prompt_lower, self.CONTEXT_PATTERNS)
        word_count = len(prompt.split())
        length_bonus = min(word_count / 50, 1) * 0.3
        
        if matches >= 3:
            score, feedback = max_score * (0.85 + length_bonus), "Rich context provided"
            suggestions = []
        elif matches >= 2:
            score, feedback = max_score * (0.6 + length_bonus), "Good context"
            suggestions = ["Add specific constraints"]
        elif matches >= 1:
            score, feedback = max_score * (0.35 + length_bonus * 0.5), "Minimal context"
            suggestions = ["Add 'Context:' section", "Explain purpose and use"]
        else:
            score, feedback = max_score * 0.1, "No context provided"
            suggestions = ["Describe the situation", "Explain intended use"]
        
        return DimensionScore("Context Depth", round(min(score, max_score), 1), max_score, feedback, suggestions)
    
    def _score_task(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['task_specificity']
        matches = self._count_patterns(prompt_lower, self.TASK_PATTERNS)
        
        action_verbs = ['write', 'create', 'generate', 'explain', 'analyze', 
                        'summarize', 'compare', 'list', 'describe', 'develop']
        verb_count = sum(1 for v in action_verbs if v in prompt_lower)
        has_numbers = bool(re.search(r'\d+', prompt))
        
        base_score = matches / 4
        verb_bonus = min(verb_count / 3, 1) * 0.2
        specificity_bonus = 0.1 if has_numbers else 0
        score_pct = min(base_score + verb_bonus + specificity_bonus, 1)
        
        if score_pct >= 0.8:
            feedback, suggestions = "Clear, specific task", []
        elif score_pct >= 0.5:
            feedback = "Task defined but could be more specific"
            suggestions = ["Add specific quantities", "Break into sub-tasks"]
        elif score_pct >= 0.25:
            feedback = "Task is vague"
            suggestions = ["Use clear action verb", "Specify exactly what you want"]
        else:
            feedback = "Task unclear or missing"
            suggestions = ["State what you want clearly", "Use '[Action] + [object] + [criteria]'"]
        
        return DimensionScore("Task Specificity", round(score_pct * max_score, 1), max_score, feedback, suggestions)
    
    def _score_format(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['output_format']
        matches = self._count_patterns(prompt_lower, self.FORMAT_PATTERNS)
        
        if matches >= 3:
            score, feedback, suggestions = max_score, "Excellent format specification", []
        elif matches >= 2:
            score, feedback = max_score * 0.7, "Good format indication"
            suggestions = ["Specify exact length"]
        elif matches >= 1:
            score, feedback = max_score * 0.4, "Basic format mentioned"
            suggestions = ["Specify length", "Indicate structure preference"]
        else:
            score, feedback = max_score * 0.15, "No format specified"
            suggestions = ["Add 'Format:' section", "Specify length and structure"]
        
        return DimensionScore("Output Format", round(score, 1), max_score, feedback, suggestions)
    
    def _score_constraints(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['constraints']
        matches = self._count_patterns(prompt_lower, self.CONSTRAINT_PATTERNS)
        
        if matches >= 3:
            score, feedback, suggestions = max_score, "Clear constraints defined", []
        elif matches >= 2:
            score, feedback = max_score * 0.7, "Some constraints specified"
            suggestions = ["Add what to avoid"]
        elif matches >= 1:
            score, feedback = max_score * 0.4, "Minimal constraints"
            suggestions = ["Add 'Avoid:' section"]
        else:
            score, feedback = max_score * 0.2, "No explicit constraints"
            suggestions = ["Consider adding boundaries"]
        
        return DimensionScore("Constraints", round(score, 1), max_score, feedback, suggestions)
    
    def _score_style(self, prompt: str, prompt_lower: str) -> DimensionScore:
        max_score = self.WEIGHTS['style_tone']
        matches = self._count_patterns(prompt_lower, self.STYLE_PATTERNS)
        
        if matches >= 3:
            score, feedback, suggestions = max_score, "Excellent style specification", []
        elif matches >= 2:
            score, feedback = max_score * 0.7, "Good style indication"
            suggestions = ["Specify audience"]
        elif matches >= 1:
            score, feedback = max_score * 0.4, "Basic style mentioned"
            suggestions = ["Specify tone", "Indicate target audience"]
        else:
            score, feedback = max_score * 0.2, "No style specified"
            suggestions = ["Add tone specification"]
        
        return DimensionScore("Style/Tone", round(score, 1), max_score, feedback, suggestions)
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 90: return "A+"
        elif score >= 85: return "A"
        elif score >= 80: return "A-"
        elif score >= 75: return "B+"
        elif score >= 70: return "B"
        elif score >= 65: return "B-"
        elif score >= 60: return "C+"
        elif score >= 55: return "C"
        elif score >= 50: return "C-"
        elif score >= 40: return "D"
        else: return "F"
    
    def _generate_feedback(self, score: float) -> str:
        if score >= 85:
            return "🌟 Excellent prompt! Well-structured with clear objectives."
        elif score >= 70:
            return "✅ Good prompt with room for minor improvements."
        elif score >= 50:
            return "⚠️ Decent prompt, but missing key elements."
        elif score >= 30:
            return "🔧 Prompt needs significant improvement."
        else:
            return "❌ Prompt is too vague. Try using a framework like CO-STAR."

# ═══════════════════════════════════════════════════════════════════════════════
# LLM OPTIMIZER (Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiOptimizer:
    """Uses Gemini to optimize prompts."""
    
    FRAMEWORKS = {
        "CO-STAR": {
            "description": "Context, Objective, Style, Tone, Audience, Response",
            "best_for": "Content creation, communication tasks",
            "template": """Apply CO-STAR:
- Context: Background information
- Objective: Specific goal
- Style: Writing style
- Tone: Emotional quality
- Audience: Target readers
- Response: Format requirements"""
        },
        "CRISPE": {
            "description": "Capacity, Role, Insight, Statement, Personality, Experiment",
            "best_for": "Technical tasks, code generation",
            "template": """Apply CRISPE:
- Capacity/Role: AI expertise
- Insight: Background context
- Statement: Specific task
- Personality: Voice/style
- Experiment: Request variations"""
        },
        "RISEN": {
            "description": "Role, Instructions, Steps, End goal, Narrowing",
            "best_for": "Complex multi-step tasks",
            "template": """Apply RISEN:
- Role: AI expertise
- Instructions: Detailed task
- Steps: Break into stages
- End goal: Success criteria
- Narrowing: Constraints"""
        },
        "RACE": {
            "description": "Role, Action, Context, Expectation",
            "best_for": "Quick, simple tasks",
            "template": """Apply RACE:
- Role: Who is the AI?
- Action: What to do?
- Context: Background
- Expectation: Output format"""
        }
    }
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def optimize(self, prompt: str, framework: str = "CO-STAR") -> Dict:
        """Optimize a prompt using the specified framework."""
        framework_info = self.FRAMEWORKS.get(framework, self.FRAMEWORKS["CO-STAR"])
        
        optimization_prompt = f"""You are an expert prompt engineer.

Rewrite this prompt using the {framework} framework:

{framework_info['template']}

ORIGINAL PROMPT:
```
{prompt}
```

Respond with JSON only:
{{
    "optimized_prompt": "<the rewritten prompt>",
    "changes_made": ["change1", "change2", ...],
    "expected_improvement": "<why this is better>"
}}"""

        try:
            response = self.model.generate_content(optimization_prompt)
            response_text = response.text.strip()
            
            # Clean JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            result = json.loads(response_text.strip())
            result['success'] = True
            result['framework'] = framework
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_gauge_chart(score: float, grade: str) -> go.Figure:
    """Create a gauge chart for the overall score."""
    
    # Determine color based on score
    if score >= 70:
        bar_color = "#22c55e"
    elif score >= 50:
        bar_color = "#f59e0b"
    else:
        bar_color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': "/100", 'font': {'size': 40, 'color': '#1f2937'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#e5e7eb"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "#1f2937", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "system-ui, -apple-system, sans-serif"}
    )
    
    return fig

def create_dimension_chart(dimensions: Dict[str, DimensionScore]) -> go.Figure:
    """Create a horizontal bar chart for dimension scores."""
    
    names = []
    scores = []
    max_scores = []
    percentages = []
    colors = []
    
    for key, dim in dimensions.items():
        names.append(dim.name)
        scores.append(dim.score)
        max_scores.append(dim.max_score)
        percentages.append(dim.percentage)
        
        if dim.percentage >= 70:
            colors.append('#22c55e')
        elif dim.percentage >= 40:
            colors.append('#f59e0b')
        else:
            colors.append('#ef4444')
    
    fig = go.Figure()
    
    # Background bars (max scores)
    fig.add_trace(go.Bar(
        y=names,
        x=max_scores,
        orientation='h',
        marker_color='#e5e7eb',
        name='Max',
        hoverinfo='skip'
    ))
    
    # Score bars
    fig.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation='h',
        marker_color=colors,
        name='Score',
        text=[f"{p:.0f}%" for p in percentages],
        textposition='inside',
        textfont=dict(color='white', size=12, family='system-ui')
    ))
    
    fig.update_layout(
        barmode='overlay',
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            range=[0, 30]
        ),
        yaxis=dict(
            showgrid=False,
            autorange='reversed'
        )
    )
    
    return fig

def create_radar_chart(dimensions: Dict[str, DimensionScore]) -> go.Figure:
    """Create a radar chart for dimension analysis."""
    
    categories = [dim.name for dim in dimensions.values()]
    values = [dim.percentage for dim in dimensions.values()]
    
    # Close the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2),
        name='Your Prompt'
    ))
    
    # Add reference line at 70%
    fig.add_trace(go.Scatterpolar(
        r=[70] * len(categories),
        theta=categories,
        line=dict(color='#22c55e', width=1, dash='dash'),
        name='Good (70%)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=350,
        margin=dict(l=60, r=60, t=40, b=60),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 PromptCraft</h1>
        <p>AI Prompt Analyzer & Optimizer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Gemini API key for optimization features"
        )
        
        st.markdown("---")
        
        # Framework selector
        st.markdown("### 📐 Optimization Framework")
        framework = st.selectbox(
            "Select framework:",
            options=["CO-STAR", "CRISPE", "RISEN", "RACE"],
            index=0
        )
        
        # Show framework info
        framework_info = GeminiOptimizer.FRAMEWORKS[framework]
        st.info(f"**{framework}**: {framework_info['description']}")
        st.caption(f"Best for: {framework_info['best_for']}")
        
        st.markdown("---")
        
        # Info section
        st.markdown("### 📚 About")
        st.markdown("""
        **PromptCraft** analyzes your prompts on 6 dimensions:
        - 📋 Task Specificity (25%)
        - 📝 Context Depth (20%)
        - 👤 Role Clarity (15%)
        - 📄 Output Format (15%)
        - 🚫 Constraints (15%)
        - 🎨 Style/Tone (10%)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ✍️ Enter Your Prompt")
        
        # Example prompts
        example_prompts = {
            "Select an example...": "",
            "❌ Bad Prompt": "Write about machine learning",
            "⚠️ Medium Prompt": "Explain how neural networks work. Use simple terms and include examples.",
            "✅ Good Prompt": """Role: You are a machine learning educator with experience teaching beginners.

Context: I'm a software developer with Python experience but no ML background.

Task: Explain how neural networks work, covering:
1. Basic architecture (neurons, layers)
2. How training works
3. A simple practical example

Format: Clear sections with headers. Include a Python code snippet.
Length: 500-700 words.

Tone: Technical but accessible. Avoid excessive jargon."""
        }
        
        selected_example = st.selectbox("Load an example:", options=list(example_prompts.keys()))
        
        prompt = st.text_area(
            "Your prompt:",
            value=example_prompts[selected_example],
            height=250,
            placeholder="Enter your prompt here...\n\nTip: Try to include role, context, task, format, and constraints.",
            label_visibility="collapsed"
        )
        
        # Analyze button
        analyze_clicked = st.button("🔍 Analyze Prompt", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if analyze_clicked and prompt.strip():
            # Initialize analyzer
            analyzer = PromptAnalyzer()
            
            with st.spinner("Analyzing your prompt..."):
                results = analyzer.analyze(prompt)
            
            # Store results in session state
            st.session_state['analysis_results'] = results
            st.session_state['original_prompt'] = prompt
            
        # Display results if available
        if 'analysis_results' in st.session_state:
            results = st.session_state['analysis_results']
            
            # Score display
            score_col, grade_col = st.columns([2, 1])
            
            with score_col:
                st.plotly_chart(
                    create_gauge_chart(results['total_score'], results['grade']),
                    use_container_width=True
                )
            
            with grade_col:
                grade = results['grade']
                grade_class = 'grade-a' if grade.startswith('A') else 'grade-b' if grade.startswith('B') else 'grade-c' if grade.startswith('C') else 'grade-d' if grade.startswith('D') else 'grade-f'
                
                st.markdown(f"""
                <div style="text-align: center; padding-top: 50px;">
                    <span class="grade-badge {grade_class}">{grade}</span>
                    <p style="margin-top: 15px; color: #6b7280;">
                        📝 {results['token_count']} tokens
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Feedback
            st.markdown(f"**{results['feedback']}**")
    
    # Tabs for detailed analysis
    if 'analysis_results' in st.session_state:
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Dimension Analysis", "🔧 Optimize", "📈 Comparison"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Score Breakdown")
                st.plotly_chart(
                    create_dimension_chart(st.session_state['analysis_results']['dimensions']),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### Radar View")
                st.plotly_chart(
                    create_radar_chart(st.session_state['analysis_results']['dimensions']),
                    use_container_width=True
                )
            
            # Detailed feedback
            st.markdown("#### 💡 Improvement Suggestions")
            
            for key, dim in st.session_state['analysis_results']['dimensions'].items():
                if dim.suggestions:
                    with st.expander(f"{dim.name}: {dim.feedback}", expanded=dim.percentage < 50):
                        for suggestion in dim.suggestions:
                            st.markdown(f"• {suggestion}")
        
        with tab2:
            if api_key:
                st.markdown(f"#### Optimize with {framework}")
                
                if st.button("✨ Generate Optimized Prompt", use_container_width=True):
                    try:
                        optimizer = GeminiOptimizer(api_key)
                        
                        with st.spinner(f"Optimizing with {framework}..."):
                            result = optimizer.optimize(st.session_state['original_prompt'], framework)
                        
                        if result.get('success'):
                            st.session_state['optimized_result'] = result
                            st.success("Optimization complete!")
                        else:
                            st.error(f"Optimization failed: {result.get('error')}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                # Show optimized result
                if 'optimized_result' in st.session_state:
                    result = st.session_state['optimized_result']
                    
                    st.markdown("#### ✨ Optimized Prompt")
                    st.markdown(f"""
                    <div class="optimized-box">
                        {result['optimized_prompt']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Changes made
                    st.markdown("#### 📝 Changes Made")
                    for change in result.get('changes_made', []):
                        st.markdown(f"• {change}")
                    
                    st.markdown("#### 💡 Expected Improvement")
                    st.info(result.get('expected_improvement', 'N/A'))
                    
                    # Copy button
                    st.code(result['optimized_prompt'], language=None)
            else:
                st.warning("⚠️ Enter your Gemini API key in the sidebar to enable optimization.")
                st.markdown("""
                **How to get an API key:**
                1. Go to [Google AI Studio](https://aistudio.google.com/)
                2. Click "Get API Key"
                3. Create a new key and paste it in the sidebar
                """)
        
        with tab3:
            if 'optimized_result' in st.session_state:
                st.markdown("#### Before vs After")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Prompt**")
                    st.text_area(
                        "original",
                        value=st.session_state['original_prompt'],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    original_score = st.session_state['analysis_results']['total_score']
                    st.metric("Score", f"{original_score}/100")
                
                with col2:
                    st.markdown("**Optimized Prompt**")
                    optimized_prompt = st.session_state['optimized_result']['optimized_prompt']
                    st.text_area(
                        "optimized",
                        value=optimized_prompt,
                        height=200,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    
                    # Analyze optimized prompt
                    analyzer = PromptAnalyzer()
                    optimized_results = analyzer.analyze(optimized_prompt)
                    new_score = optimized_results['total_score']
                    improvement = new_score - original_score
                    
                    st.metric(
                        "Score",
                        f"{new_score}/100",
                        delta=f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
                    )
                
                # Comparison chart
                st.markdown("#### Score Comparison")
                
                fig = go.Figure()
                
                dimensions = list(st.session_state['analysis_results']['dimensions'].keys())
                original_scores = [st.session_state['analysis_results']['dimensions'][d].percentage for d in dimensions]
                optimized_scores = [optimized_results['dimensions'][d].percentage for d in dimensions]
                
                fig.add_trace(go.Bar(
                    name='Original',
                    x=[d.replace('_', ' ').title() for d in dimensions],
                    y=original_scores,
                    marker_color='#94a3b8'
                ))
                
                fig.add_trace(go.Bar(
                    name='Optimized',
                    x=[d.replace('_', ' ').title() for d in dimensions],
                    y=optimized_scores,
                    marker_color='#22c55e'
                ))
                
                fig.update_layout(
                    barmode='group',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="Score %",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Optimize your prompt first to see the comparison.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 0.875rem;">
        Built with ❤️ for INFO 7390 | Powered by Gemini AI<br>
        <em>Created with AI assistance (Claude by Anthropic)</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
