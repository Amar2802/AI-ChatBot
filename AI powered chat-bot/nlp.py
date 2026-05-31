from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from config import GEN_MODEL, SIM_THRESHOLD, MAX_TURNS_MEMORY

# Load tokenizer and model for CausalLM
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL,
        torch_dtype=torch.float32
    ).to(device)

SYSTEM_STYLE = (
    "You are a professional, helpful support and general assistant. "
    "Use the provided FAQ context (if any) to answer questions factually and contextually. "
    "If the context matches the user's question, base your response directly on it. "
    "If the query is a greeting or chit-chat, reply warmly and professionally. "
    "If the query is a technical or programming question (e.g. coding help, explanations, concepts) "
    "that is not in the FAQ context, answer it using your own pre-trained knowledge. Provide detailed explanations, "
    "examples, clean code blocks with syntax highlighting (e.g. ```python), and best practices. "
    "If the user's request is extremely short, vague, or ambiguous (e.g. just 'how to reset', 'help', or incomplete thoughts), "
    "do not guess; instead, politely ask clarifying questions to understand what they are trying to achieve."
)

def generate_answer(user_query: str, history: List[Tuple[str, str]], retrieved, max_new_tokens: int = 512) -> str:
    # Filter retrieved FAQ candidates to only keep those with similarity score >= 0.35
    valid_retrieved = [r for r in retrieved if len(r) >= 4 and r[3] >= 0.35]
    
    # Construct System Prompt based on whether we have relevant FAQ context
    if valid_retrieved:
        context_blocks = []
        for _id, q, a, score in valid_retrieved:
            context_blocks.append(f"FAQ Question: {q}\nFAQ Answer: {a}")
        context_text = "\n\n".join(context_blocks)
        system_prompt = (
            f"{SYSTEM_STYLE}\n\n"
            f"Here is the relevant FAQ Context from our database:\n"
            f"{context_text}\n\n"
            "Use this context as your primary source of truth if the user's question matches it."
        )
    else:
        system_prompt = SYSTEM_STYLE
        
    # Trim and construct conversational message history
    # Excluding the last user query which will be added explicitly
    trimmed = history[:-1][-MAX_TURNS_MEMORY:]
    
    messages = [{"role": "system", "content": system_prompt}]
    for role, text in trimmed:
        messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": user_query})
    
    try:
        # Apply the chat template to format conversation into Qwen's expected structure
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Generate response tokens
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract only the newly generated tokens (excluding prompt) and decode
        generated_ids = outputs[0][len(input_ids[0]):]
        ans = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Response Validation: handle edge cases where output might be empty or problematic
        if not ans:
            return "I'm sorry, I couldn't generate an answer. Please try rephrasing your question."
            
        return ans
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        return "I'm sorry, I encountered an issue processing your request. Please try again or contact support."

def choose_response(user_query: str, retrieved):
    """
    Heuristic: if top similarity clears the threshold (>= 0.55) and is a specific FAQ query,
    return its answer directly for zero latency.
    Otherwise, fall back to the generator with retrieved context as grounding.
    """
    if not retrieved:
        return None, 0.0, "no_hits"
    
    top = retrieved[0]
    _, _q, _a, score = top
    
    # If high similarity FAQ match, return directly
    if score >= SIM_THRESHOLD:
        return _a, score, "direct"
        
    return None, score, "fallback"
