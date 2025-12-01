from dotenv import load_dotenv
load_dotenv()
import os
import httpx
from mcp.server.fastmcp import FastMCP
import asyncio
import json

API_KEY = os.getenv("OPENROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": "http://localhost",
    "X-Title": "LLMspeaks",
    "Content-Type": "application/json"
}

mcp = FastMCP()

@mcp.tool()
async def call_grok(prompt: str, system: str = None) -> str:
    data = {
        "model": "x-ai/grok-4.1-fast:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        data["messages"].insert(0, {"role": "system", "content": system})
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(URL, headers=HEADERS, json=data)
        return json.dumps(r.json())

@mcp.tool()
async def call_mistral(prompt: str, system: str = None) -> str:
    data = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        data["messages"].insert(0, {"role": "system", "content": system})
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(URL, headers=HEADERS, json=data)
        return json.dumps(r.json())

@mcp.tool()
async def debate_llm(prompt: str) -> str:
    rounds = 4
    
    grok_system = "You are a critical thinker focused on first principles, evidence, and challenging assumptions. Be concise but thorough."
    mistral_system = "You are an analytical thinker focused on practical implications, counterarguments, and real-world constraints. Be concise but thorough."
    
    debate_history = []
    
    grok_prompt = f"Analyze this question from your perspective:\n{prompt}"
    mistral_prompt = f"Analyze this question from your perspective:\n{prompt}"
    
    for i in range(rounds):
        grok_resp_str, mistral_resp_str = await asyncio.gather(
            call_grok(grok_prompt, grok_system),
            call_mistral(mistral_prompt, mistral_system)
        )
        
        grok_resp = json.loads(grok_resp_str)
        mistral_resp = json.loads(mistral_resp_str)
        
        if "choices" not in grok_resp:
            return f"Grok API error: {grok_resp}"
        if "choices" not in mistral_resp:
            return f"Mistral API error: {mistral_resp}"
        
        grok_text = grok_resp["choices"][0]["message"]["content"]
        mistral_text = mistral_resp["choices"][0]["message"]["content"]
        
        debate_history.append(f"Round {i+1} - Grok: {grok_text}")
        debate_history.append(f"Round {i+1} - Mistral: {mistral_text}")
        
        grok_prompt = f"Mistral responded:\n{mistral_text}\n\nAddress their points and strengthen your argument:"
        mistral_prompt = f"Grok responded:\n{grok_text}\n\nAddress their points and strengthen your argument:"
    
    synthesis_prompt = f"""Original question: {prompt}

Complete debate transcript:
{chr(10).join(debate_history)}

Your task: Synthesize the BEST possible answer by:
1. Extracting strongest evidence and reasoning from both sides
2. Resolving contradictions by evaluating quality of arguments
3. Identifying where models agree (likely correct) vs disagree (requires nuance)
4. Providing actionable, clear conclusions

Be direct, precise, and cut through noise. No hedging."""

    final_str = await call_grok(synthesis_prompt, "You are an expert synthesizer. Produce the most accurate, useful answer possible.")
    final_resp = json.loads(final_str)
    
    if "choices" not in final_resp:
        return f"Synthesis error: {final_resp}"
    
    return final_resp["choices"][0]["message"]["content"]

def main():
    if not API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()