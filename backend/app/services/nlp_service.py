from typing import List, Tuple, Optional
from app.core.config import settings

# Lazy loading variables for local model
_local_tokenizer = None
_local_model = None
_openai_failed = False
_gemini_failed = False

def _get_local_model_and_tokenizer():
    global _local_tokenizer, _local_model
    if _local_model is None or _local_tokenizer is None:
        print(f"Loading local model and tokenizer for '{settings.LOCAL_GEN_MODEL}' on CPU/GPU...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Optimize CPU threads for PyTorch to avoid thread thrashing/high CPU wait times
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)
            
        _local_tokenizer = AutoTokenizer.from_pretrained(settings.LOCAL_GEN_MODEL)
        if _local_tokenizer.pad_token_id is None:
            _local_tokenizer.pad_token_id = _local_tokenizer.eos_token_id
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            _local_model = AutoModelForCausalLM.from_pretrained(
                settings.LOCAL_GEN_MODEL,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            _local_model = AutoModelForCausalLM.from_pretrained(
                settings.LOCAL_GEN_MODEL,
                torch_dtype=torch.float32
            ).to(device)
    return _local_model, _local_tokenizer

SYSTEM_STYLE = (
    "You are a professional, helpful support and general assistant. "
    "Respond in a clear, extremely concise manner. Keep explanations brief and direct unless detailed code is requested. "
    "Use the provided FAQ context (if any) to answer questions factually and contextually. "
    "If the context matches the user's question, base your response directly on it. "
    "If the query is a greeting or chit-chat, reply warmly and professionally. "
    "If the query is a technical or programming question (e.g. coding help, explanations, concepts) "
    "that is not in the FAQ context, answer it using your own pre-trained knowledge. Provide detailed explanations, "
    "examples, clean code blocks with syntax highlighting (e.g. ```python), and best practices. "
    "If the user's request is extremely short, vague, or ambiguous (e.g. just 'how to reset', 'help', or incomplete thoughts), "
    "do not guess; instead, politely ask clarifying questions to understand what they are trying to achieve."
)

def choose_response(user_query: str, retrieved) -> Tuple[Optional[str], float, str]:
    """
    Decide whether to return an FAQ answer directly or fall back to the LLM.
    """
    if not retrieved:
        return None, 0.0, "no_hits"

    top = retrieved[0]
    _, _q, _a, score = top
    second_score = retrieved[1][3] if len(retrieved) > 1 else 0.0
    margin = score - second_score

    # Strong match: return stored FAQ answer verbatim
    if score >= settings.SIM_THRESHOLD:
        return _a, score, "direct"

    # Clear winner over other FAQs (handles paraphrases and short queries)
    if score >= settings.FAQ_CONTEXT_MIN and margin >= 0.08:
        return _a, score, "direct"

    return None, score, "fallback"

def generate_answer_stream(user_query: str, history: List[Tuple[str, str]], retrieved, max_new_tokens: int = None):
    """
    Generates LLM text stream using OpenAI, Gemini, or local models.
    """
    global _openai_failed, _gemini_failed
    tokens_limit = max_new_tokens or settings.MAX_NEW_TOKENS
    valid_retrieved = [r for r in retrieved if len(r) >= 4 and r[3] >= settings.FAQ_CONTEXT_MIN]
    
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
        
    # Prior turns only (current query is appended inside generation routines)
    trimmed = history[:-1][-(settings.MAX_TURNS_MEMORY * 2):]

    # 1. If OpenAI API key is available, use OpenAI API
    if settings.OPENAI_API_KEY and not settings.USE_LOCAL_LLM and not _openai_failed:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            messages = [{"role": "system", "content": system_prompt}]
            for role, text in trimmed:
                messages.append({"role": role, "content": text})
            messages.append({"role": "user", "content": user_query})
            
            response = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=tokens_limit,
                stream=True
            )
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            return

        except Exception as e:
            print(f"Error in OpenAI API generation: {e}. Disabling OpenAI API fallback.")
            _openai_failed = True

    # 2. If Gemini API key is available, use Gemini API
    if settings.GEMINI_API_KEY and not settings.USE_LOCAL_LLM and not _gemini_failed:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
            contents = []
            for role, text in trimmed:
                gemini_role = "user" if role == "user" else "model"
                contents.append(
                    types.Content(
                        role=gemini_role,
                        parts=[types.Part.from_text(text=text)]
                    )
                )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)]
                )
            )
            
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
                max_output_tokens=tokens_limit,
            )
            
            response = client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=contents,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return

        except Exception as e:
            print(f"Error in Gemini API generation: {e}. Disabling Gemini API fallback.")
            _gemini_failed = True

    # 3. Local LLM fallback
    try:
        model, tokenizer = _get_local_model_and_tokenizer()
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        # Ensure threads are set correctly before execution starts
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)

        messages = [{"role": "system", "content": system_prompt}]
        for role, text in trimmed:
            messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_query})
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Greedy decoding (do_sample=False) is significantly faster on CPU
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=tokens_limit,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
            
    except Exception as e:
        print(f"Error in local generate_answer: {e}")
        yield "I'm sorry, I encountered an issue processing your request. Please try again or contact support."

def generate_answer(user_query: str, history: List[Tuple[str, str]], retrieved, max_new_tokens: int = None) -> str:
    """
    Synchronous wrapper consuming the generator.
    """
    return "".join(generate_answer_stream(user_query, history, retrieved, max_new_tokens))
