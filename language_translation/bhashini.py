import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Mapping Hindi → Target Scripts (transliteration)
lang_script_map = {
    "hin_Deva": ("en","hi"),   # Hindi
    "tam_Taml": ("hi","ta"),   # Tamil
    "tel_Telu": ("hi","te"),   # Telugu
    "kan_Knda": ("hi","kn"),   # Kannada
    "mal_Mlym": ("hi","ml"),   # Malayalam
    "ben_Beng": ("hi","bn"),   # Bengali
    "pan_Guru": ("en","pa"),   # Punjabi (Gurmukhi)
    "mar_Deva": ("hi","mr"),   # Marathi
    "guj_Gujr": ("hi","gu"),   # Gujarati
    "ory_Orya": ("hi","or"),   # Odia
    "asm_Beng": ("hi","as"),   # Assamese
    "san_Deva": ("hi","sa"),   # Sanskrit
    "npi_Deva": ("hi","ne"),   # Nepali
    "gom_Deva": ("hi","ks"),   # Konkani (Devanagari)
    "kas_Arab": ("hi","ur"),   # Kashmiri Arabic → Urdu-style
    "kas_Deva": ("hi","ks"),   # Kashmiri Devanagari
    "snd_Arab": ("hi","ur"),   # Sindhi Arabic
    "snd_Deva": ("hi","sa"),   # Sindhi Devanagari
    "urd_Arab": ("hi","ur")    # Urdu
}

def translate(text: str, src_lang="eng_Latn", tgt_lang="hin_Deva"):
    # 1️⃣ Prepare input
    input_text = f"{src_lang} {tgt_lang} {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # 2️⃣ Stable config
    gen_cfg = GenerationConfig(
        use_cache=False,
        do_sample=False,
        max_new_tokens=2048,
        num_beams=1
    )

    # 3️⃣ Generate text
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_cfg)

    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 4️⃣ Script conversion (only if mapping exists)
    if tgt_lang in lang_script_map:
        src_code, tgt_code = lang_script_map[tgt_lang]
        translated = UnicodeIndicTransliterator.transliterate(translated, src_code, tgt_code)

    return translated

sent = """hello, how are you? this is a test sentence for translation."""

print("Hindi   :", translate(sent, "eng_Latn", "hin_Deva"))
# print("Punjabi :", translate(sent, "eng_Latn", "pan_Guru"))

