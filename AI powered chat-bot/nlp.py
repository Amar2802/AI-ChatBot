from typing import List, Tuple
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import GEN_MODEL, SIM_THRESHOLD, MAX_TURNS_MEMORY
# Load generator (FLAN-T5 works well for instruction following)
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
SYSTEM_STYLE = (
	"You are a helpful, concise support chatbot."
	" If context is provided as FAQ entries, strictly answer using it."
	" If the answer is unknown, say you don't know and suggest contacting support."
)
def build_prompt(user_query: str, history: List[Tuple[str,str]], retrieved:
	List[Tuple[int,str,str,float]]
) -> str:
	context_blocks = []
	for _id, q, a, score in retrieved:
		context_blocks.append(f"Q: {q}\nA: {a}")
	context_text = "\n\n".join(context_blocks)
	# Keep short history 
	trimmed = history[-MAX_TURNS_MEMORY:]
	hist_text = "\n".join([f"{r.upper()}: {t}" for r, t in trimmed])
	prompt = (
		f"{SYSTEM_STYLE}\n\n"
		f"Context:\n{context_text if context_text else 'No context.'}\n\n"
		f"Conversation so far:\n{hist_text if hist_text else 'None'}\n\n"
		f"USER: {user_query}\n"
		f"ASSISTANT:"
	)
	return prompt
def generate_answer(user_query: str, history: List[Tuple[str,str]],
retrieved, max_new_tokens: int = 192) -> str:
	prompt = build_prompt(user_query, history, retrieved)
	inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
	outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
	return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
def choose_response(user_query: str, retrieved):
	"""Heuristic: if top similarity clears threshold, return its answer
	directly.
	Else, use generator with retrieved context as grounding."""
	if not retrieved:
		return None, 0.0, "no_hits"
	top = retrieved[0]
	_, _q, _a, score = top
	if score >= SIM_THRESHOLD:
		return _a, score, "direct"
	return None, score, "fallback"
