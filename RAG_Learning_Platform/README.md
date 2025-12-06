## 🛠️ Setup Instructions

### 1. Get API Key
This project uses Google's Gemini 2.5 Flash model.

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your key

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_actual_key_here
```

### 4. Test Your Setup
```python
python test_setup.py
```

You should see: "✅ Setup successful!"
```
