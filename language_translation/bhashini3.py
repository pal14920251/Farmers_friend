
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"

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

tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

model.to(device)
model.eval()

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
        output_ids = model.generate(
            **inputs,
            generation_config=GEN_CFG
        )

    translated = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

    if tgt_lang in lang_script_map:
        src_code, tgt_code = lang_script_map[tgt_lang]
        translated = UnicodeIndicTransliterator.transliterate(
            translated, src_code, tgt_code
        )

    return translated

def chunk_text(text, max_chars=500):
    """
    Safely split large text into smaller character-based chunks.
    This method is 100% compatible with IndicTrans2.
    """
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
        translated_chunks.append(
            translate(ch, src_lang=src_lang, tgt_lang=tgt_lang)
        )

    return "\n".join(translated_chunks)

sent = prompt
output = translate_large_text(
    sent,
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
    max_chars=400
)

# save output to text file 
with open("translated_report_hindi.txt", "w", encoding="utf-8") as f:
    f.write(output)




