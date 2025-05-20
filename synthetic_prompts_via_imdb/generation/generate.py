from llama_cpp import Llama
from icecream import ic
import random
import json
import os
import re
import time
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the quantised model weights (adjust if your location is different)
MODEL_PATH = "/projects/constitutional_classifier/synthetic_prompts_via_imdb/mistral/model_weights/Ministral-8B-Instruct-2410-Q6_K.gguf"
# GPU to use (set to an empty string to force CPU inference)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -----------------------------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------------------------
print("Starting model loading …")
llm = Llama.from_pretrained(
    repo_id="bartowski/Ministral-8B-Instruct-2410-GGUF",
    filename="Ministral-8B-Instruct-2410-Q6_K.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,  # ‑1 = place as many layers on‑GPU as will fit
    verbose=False,
    use_mlock=False,
    flash_attn=False,
    echo=False,
)
print("Model ready ✔")

# -----------------------------------------------------------------------------
# Prompt generation helpers
# -----------------------------------------------------------------------------

def generate_prompts(
    llm,
    topic: str = "medicine",
    temperature_q: float = 1.2,
    temperature_p: float = 0.7,
):
    """Generate (questions, prompts) pairs in the medical domain.

    * *questions* imitate real user queries
    * *prompts* are re‑written instructions for the assistant
    """
    # 1️⃣  Create user‑style questions
    questions_resp = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a synthetic data generator. Generate authentic‑sounding "
                    "questions that a diverse set of users might ask an AI assistant "
                    "in the medical domain. Vary complexity, length, style, and "
                    "specificity."
                ),
            },
            {
                "role": "user",
                "content": f"""Generate 3 diverse questions about {topic}. Include a mix of:
- Factual questions seeking medical information (definitions, causes, symptoms, treatment options, etc.)
- Opinion‑based questions asking for analysis or professional judgement (comparisons of treatment efficacy, lifestyle recommendations, medical ethics, health policy)
- Creative or hypothetical questions (future of medicine, innovative therapies, \"what‑if\" medical scenarios)

Vary question styles (direct questions, requests for explanations, comparisons, case‑style scenarios). Keep everything **strictly within the medical domain**.""",
            },
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
            },
        },
        temperature=temperature_q,
        max_tokens=1024,
        seed=random.randint(0, 1_000_000),
    )

    try:
        questions = json.loads(questions_resp["choices"][0]["message"]["content"]) [
            "questions"
        ]
    except (KeyError, json.JSONDecodeError):
        return None, None

    # 2️⃣  Re‑write as assistant prompts (still medical‑only)
    prompts_resp = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a synthetic data generator. Create diverse, natural‑sounding "
                    "prompts that reflect how humans interact with AI assistants. Ensure "
                    "variety in tone, complexity, and format. The most important thing is "
                    "that the prompts must remain **strictly within the medical domain**."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on the following user questions, generate 3 AI‑assistant prompts "
                    "of different styles.\n\nUser questions:\n‑ "
                    + "\n‑ ".join(questions)
                    + "\n\nGuidelines:\n1. Use only medical context, terminology, and scenarios.\n2. Vary length (short, medium, detailed).\n3. Vary tone (casual, academic, empathetic).\n4. Include different request types (explanations, comparisons, creative tasks).\n5. Do NOT include content outside of medicine.\n\nReturn a JSON object with an array 'prompts'."
                ),
            },
        ],
        response_format={
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "prompts": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                        "maxItems": 3,
                    }
                },
                "required": ["prompts"],
            },
        },
        temperature=temperature_p,
        max_tokens=1024,
        seed=random.randint(0, 1_000_000),
    )

    try:
        prompts = json.loads(prompts_resp["choices"][0]["message"]["content"]) [
            "prompts"
        ]
    except (KeyError, json.JSONDecodeError):
        return None, None

    return questions, prompts

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """Convert topic names to safe filenames."""
    return re.sub(r"[^\w\-]+", "_", name)

# -----------------------------------------------------------------------------
# Main execution loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    topics = [
        "cardiology",
        "neurology",
        "pediatrics",
        "oncology",
        "public health",
    ]

    start_time = time.time()

    # Generate approximately 100k prompts/questions in total (3 pairs per topic per loop)
    for i in tqdm(range(100_000 // 3)):
        topic = topics[i % len(topics)]

        questions, prompts = generate_prompts(
            llm,
            topic=topic,
            temperature_q=2.5,  # hotter for more diverse questions
            temperature_p=0.4,  # cooler for more controlled prompts
        )
        if questions is None:
            continue  # skip iteration if generation failed

        fn_base = safe_filename(topic)
        os.makedirs("data", exist_ok=True)

        with open(f"data/{fn_base}_prompts.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(prompts) + "\n")

        with open(f"data/{fn_base}_questions.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(questions) + "\n")

    total_time = time.time() - start_time
    ic(f"Time taken to generate prompts: {total_time:.2f} seconds")
