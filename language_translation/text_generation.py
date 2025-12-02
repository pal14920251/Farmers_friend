import json
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load HF API Token
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ----------------------
# LLM SETUP
# ----------------------
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_llm = HuggingFaceEndpoint(
    repo_id=MODEL_NAME,
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.9,
)

chatmodel = ChatHuggingFace(
    llm=hf_llm,
    temperature=0.1,
)

# =================================================================
# 1️⃣ LOAD ANY PROMPT TEMPLATE (supports multiple JSON prompt files)
# =================================================================
def load_prompt_template(json_path: str, key: str) -> PromptTemplate:
    """
    Loads a prompt template from any JSON file.
    JSON must contain:
    {
        "template_key": {
            "template": "...",
            "input_variables": ["x","y"]
        }
    }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompt file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {json_path}")

    template_block = data[key]
    template_str = template_block["template"]
    input_vars = template_block["input_variables"]

    return PromptTemplate(template=template_str, input_variables=input_vars)


# =================================================================
# 2️⃣ LOAD SESSION RESULTS (crop + irrigation + weather info)
# =================================================================
def load_session_results(path="/data1/home/anumalas/GENAI-PROJECT/UI/session_results.json"):
    if not os.path.exists(path):
        raise FileNotFoundError("session_results.json not found. Run recommendation first.")

    with open(path, "r") as f:
        return json.load(f)


# =================================================================
# 3️⃣ UNIVERSAL REPORT GENERATOR (works for ANY prompt template)
# =================================================================
def generate_report_from_template(prompt_file, template_key, session_data):
    prompt_template = load_prompt_template(prompt_file, template_key)
    formatted_prompt = prompt_template.format(**session_data)
    result = chatmodel.invoke(formatted_prompt)
    return result.content.strip()


# =================================================================
# Example: Generate final crop report using template file
# =================================================================
def generate_comprehensive_crop_report():
    session = load_session_results()

    report = generate_report_from_template(
        prompt_file="/data1/home/anumalas/GENAI-PROJECT/UI/language_translation/prompt_template.json",
        template_key="crop_report_prompt",
        session_data=session
    )

    return report


# RUN LOCAL REPORT GENERATION
report = generate_comprehensive_crop_report()


# =================================================================
# 4️⃣ TRANSLATION SYSTEM (IndicTrans2)
# =================================================================

lang_script_map = {
    "hin_Deva": ("hi","hi"),
    "tam_Taml": ("hi","ta"),
    "tel_Telu": ("hi","te"),
    "kan_Knda": ("hi","kn"),
    "mal_Mlym": ("hi","ml"),
    "ben_Beng": ("hi","bn"),
    "pan_Guru": ("hi","pa"),
    "mar_Deva": ("hi","mr"),
    "guj_Gujr": ("hi","gu"),
    "ory_Orya": ("hi","or"),
    "asm_Beng": ("hi","as"),
    "san_Deva": ("hi","sa"),
    "npi_Deva": ("hi","ne"),
    "gom_Deva": ("hi","ks"),
    "kas_Arab": ("hi","ur"),
    "kas_Deva": ("hi","ks"),
    "snd_Arab": ("hi","ur"),
    "snd_Deva": ("hi","sa"),
    "urd_Arab": ("hi","ur")
}

GEN_CFG = GenerationConfig(
    use_cache=False,
    do_sample=False,
    max_new_tokens=400,
    num_beams=1
)

model_name = "ai4bharat/indictrans2-en-indic-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model.to(device)
model.eval()


# =================================================================
# TRANSLATION HELPERS
# =================================================================
def translate(text: str, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    text = text.strip()

    input_text = f"{src_lang} {tgt_lang} {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=False,
        truncation=True
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=GEN_CFG)

    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Script transliteration
    if tgt_lang in lang_script_map:
        src_code, tgt_code = lang_script_map[tgt_lang]
        translated = UnicodeIndicTransliterator.transliterate(translated, src_code, tgt_code)

    return translated


def chunk_text(text, max_chars=500):
    text = text.strip()
    chunks = []

    while len(text) > max_chars:
        split_pos = text.rfind(". ", 0, max_chars)
        if split_pos == -1:
            split_pos = max_chars
        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()

    if text:
        chunks.append(text)

    return chunks


def translate_large_text(full_text, src_lang="eng_Latn", tgt_lang="hin_Deva", max_chars=500):
    chunks = chunk_text(full_text, max_chars=max_chars)

    print(f"[INFO] Total chunks created: {len(chunks)}")

    translated_chunks = []
    for i, ch in enumerate(chunks, start=1):
        print(f"[INFO] Translating chunk {i}/{len(chunks)} ...")
        translated_chunks.append(translate(ch, src_lang=src_lang, tgt_lang=tgt_lang))

    return "\n".join(translated_chunks)
