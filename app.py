import streamlit as st
from clova_ocr import get_ocr_result, get_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import json


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


@st.cache_resource()
def load_tokenizer_model_config(model_name_or_path: str):
    with st.spinner("Loading Model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        config = GenerationConfig.from_pretrained("t5-base")
    return tokenizer, model, config


def main():
    st.title("OCR Demo")
    _URL_KEY = load_json("data/clova_key.json")
    tokenizer, model, gen_config = load_tokenizer_model_config("psyche/KoT5")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    _prompt = st.selectbox(
        "사용자 양식",
        [
            "일자,세대명,위치,공종,부위,유형,내용",
            "일자,세대명,위치,부위,결함,상세내용",
            "일자,세대명,위치,부위,공종,하자내용",
            "일자,세대명,하자내용",
            "일자,세대명,하자유형,발생부위,하자내용",
            "일자,세대명,위치,공종,하자내용",
            "일자,세대명,위치,공종,하자유형,상세내용"
        ]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        st.write("Result:")
        result = get_ocr_result(
            uploaded_file,
            secret_key=_URL_KEY["key"],
            api_url=_URL_KEY["url"]
        )

        ocr_text = get_text(result)
        input_text = f"{_prompt} {ocr_text}"
        inputs = tokenizer(input_text, return_tensors="pt")
        gen_config.max_length = 128
        output = model.generate(**inputs, generation_config=gen_config)
        st.write(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
        st.write("Raw Text:")
        st.write(ocr_text)
        st.write("Input Text:")
        st.write(input_text)
        st.write("Output Text:")
        st.write(tokenizer.batch_decode(output, skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()