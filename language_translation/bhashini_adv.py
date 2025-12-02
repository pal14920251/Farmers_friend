import os
import json
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
import textwrap

# ============================================================
#                        LOAD ENV + LLM
# ============================================================
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")



MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ============================================================
#               CHUNKING FUNCTION FOR TRANSLATION
# ============================================================

def chunk_text(text, max_chars=2500):
    """
    Splits the input text into smaller chunks so that translation
    does not break or truncate.
    """
    chunks = textwrap.wrap(text, max_chars, break_long_words=False, replace_whitespace=False)
    return chunks


def translate_chunk(chunk_text, target_language_code):

    inputs = tokenizer(chunk_text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[target_language_code],
            max_length=2048,
        )

    translated = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    return translated


def translate_large_text(full_text, target_language_code="hin_Deva"):
    """
    Breaks text into chunks → translates each → merges output.
    """

    chunks = chunk_text(full_text)
    final_output = []

    print(f"[INFO] Total chunks created: {len(chunks)}")

    for i, ch in enumerate(chunks, start=1):
        print(f"[INFO] Translating chunk {i}/{len(chunks)} ...")
        translated = translate_chunk(ch, target_language_code)
        final_output.append(translated)

    return "\n".join(final_output)


# ============================================================
#             MAIN FUNCTION USED BY YOUR PIPELINE
# ============================================================

def generate_final_output(target_language_code, **kwargs):
    """
    Receives the final report text → applies robust translation.
    """

    # -----------------------
    # 1. Generate your English output (unchanged)
    # -----------------------
    template = """
    You are a domain expert. Create a final detailed report based on:

    District: {district}
    Type: {type}
    Notes: {notes}

    Make the output high-quality and informative.
    """

    prompt = PromptTemplate(
        input_variables=["district", "type", "notes"],
        template=template
    )

    final_text = prompt.format(**kwargs)

    # -----------------------
    # 2. Translate in chunks
    # -----------------------
    translated = translate_large_text(final_text, target_language_code)

    return translated


# ============================================================
#                   TESTING / SAMPLE RUN
# ============================================================

if __name__ == "__main__":

    sample_data = {
        "district": "Raebareli",
        "type": "Agriculture",
        "notes": "Wheat production expected to increase this year due to better rainfall"
    }

    target_lang = "hin_Deva"   # Change for other languages

    final_output = generate_final_output(
        target_language_code=target_lang,
        **sample_data
    )

    print("\n=========== FINAL OUTPUT ===========\n")
    print(final_output)
    print("\n====================================\n")