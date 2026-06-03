from typing import List, Tuple
from config import (
    GEN_MODEL,
    SIM_THRESHOLD,
    FAQ_CONTEXT_MIN,
    MAX_TURNS_MEMORY,
    GEMINI_API_KEY,
    USE_LOCAL_LLM
)

# Lazy loading variables for local model
_local_tokenizer = None
_local_model = None

def _get_local_model_and_tokenizer():
    global _local_tokenizer, _local_model
    if _local_model is None or _local_tokenizer is None:
        print(f"Loading local model and tokenizer for '{GEN_MODEL}' on CPU/GPU...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        _local_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
        if _local_tokenizer.pad_token_id is None:
            _local_tokenizer.pad_token_id = _local_tokenizer.eos_token_id
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            _local_model = AutoModelForCausalLM.from_pretrained(
                GEN_MODEL,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            _local_model = AutoModelForCausalLM.from_pretrained(
                GEN_MODEL,
                torch_dtype=torch.float32
            ).to(device)
    return _local_model, _local_tokenizer

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
    valid_retrieved = [r for r in retrieved if len(r) >= 4 and r[3] >= FAQ_CONTEXT_MIN]
    
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
            "If the user's question matches any FAQ above, answer using that FAQ's answer "
            "accurately and concisely. Do not invent policies or details that are not in the FAQ."
        )
    else:
        system_prompt = SYSTEM_STYLE
        
    # Prior turns only (current user message is appended below)
    trimmed = history[:-1][-(MAX_TURNS_MEMORY * 2):]

    # If API key is available and USE_LOCAL_LLM is false, use Gemini API
    if GEMINI_API_KEY and not USE_LOCAL_LLM:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GEMINI_API_KEY)
            
            contents = []
            for role, text in trimmed:
                gemini_role = "user" if role == "user" else "model"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)]
                    )
                )
            # Append current query
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)]
                )
            )
            
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=max_new_tokens,
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=config,
            )
            
            ans = response.text.strip() if response.text else ""
            if not ans:
                return "I'm sorry, I couldn't generate an answer from the API. Please try rephrasing your question."
            return ans

        except Exception as e:
            print(f"Error in Gemini API generation: {e}. Falling back to local model if possible...")
            # Fall through to local model

    # Local LLM fallback
    try:
        model, tokenizer = _get_local_model_and_tokenizer()
        import torch

        messages = [{"role": "system", "content": system_prompt}]
        for role, text in trimmed:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_query})
        
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
            return "I'm sorry, I couldn't generate an answer locally. Please try rephrasing your question."
            
        return ans
    except Exception as e:
        print(f"Error in local generate_answer: {e}")
        return "I'm sorry, I encountered an issue processing your request. Please try again or contact support."


def choose_response(user_query: str, retrieved):
    """
    Return the FAQ answer directly when retrieval is confident enough.
    Otherwise fall back to the generator with retrieved FAQ context.
    """
    if not retrieved:
        return None, 0.0, "no_hits"

    top = retrieved[0]
    _, _q, _a, score = top
    second_score = retrieved[1][3] if len(retrieved) > 1 else 0.0
    margin = score - second_score

    # Strong match: return stored FAQ answer verbatim
    if score >= SIM_THRESHOLD:
        return _a, score, "direct"

    # Clear winner over other FAQs (handles paraphrases and short queries)
    if score >= FAQ_CONTEXT_MIN and margin >= 0.08:
        return _a, score, "direct"

    return None, score, "fallback"
