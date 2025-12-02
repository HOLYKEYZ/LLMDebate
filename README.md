# LLM Debate Engine

Multi-model debate system using Grok and Mistral via OpenRouter to produce synthesized answers.

## Setup
```bash
pip install python-dotenv httpx mcp
```

Create `.env`:
```
OPENROUTER_API_KEY=your_key_here
```

## Usage
```bash
python server.py
```

### Tools

- `call_grok(prompt)` - Query Grok
- `call_mistral(prompt)` - Query Mistral  
- `debate_llm(prompt)` - Run 4-round debate, get synthesized answer

## How It Works

1. Both models analyze question independently
2. 4 rounds of back-and-forth argumentation
3. Final synthesis extracts best reasoning from both
4. Returns single optimized answer