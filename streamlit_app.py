import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import os

class ImageCaptionPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        self.translator_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        self.translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(self.device)

    def generate_caption(self, image, prompt=None, language="Русский"):
        image = image.convert("RGB")
        inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.blip_model.generate(**inputs, max_length=200, num_beams=4)
            english_caption = self.blip_processor.decode(output_ids[0], skip_special_tokens=True)
        if language == "Русский":
            translated_inputs = self.translator_tokenizer(english_caption, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                translated_ids = self.translator_model.generate(**translated_inputs, max_length=200, num_beams=4)
                russian_caption = self.translator_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            return english_caption, russian_caption
        return english_caption, None

def main():
    st.title("Генерация подписей к изображениям")

    pipeline = ImageCaptionPipeline()
    
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png", "jpeg"])
    language = st.selectbox("Выберите язык:", ["Русский", "English"])
    prompt_options = [
        "Опиши сцену в стиле сказки.",
        "Опиши сцену как новостной репортаж.",
        "Опиши сцену в поэтическом стиле."
    ]
    prompt_choice = st.selectbox("Выберите стиль описания или введите свой:", 
                                 ["Выберите вариант"] + prompt_options + ["Свой промпт"])
    custom_prompt = None
    if prompt_choice == "Свой промпт":
        custom_prompt = st.text_input("Введите свой промпт:")
    elif prompt_choice != "Выберите вариант":
        custom_prompt = prompt_choice
    
    enable_audio = st.checkbox("Включить озвучку", value=False)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        if st.button("Сгенерировать подпись"):
            english_caption, russian_caption = pipeline.generate_caption(image, prompt=custom_prompt, language=language)
            st.write(f"**English Caption**: {english_caption}")
            if russian_caption:
                st.write(f"**Русское описание**: {russian_caption}")
            
            if enable_audio:
                text_to_speak = russian_caption if language == "Русский" else english_caption
                lang_code = "ru" if language == "Русский" else "en"
                tts = gTTS(text=text_to_speak, lang=lang_code)
                audio_file = "caption_audio.mp3"
                tts.save(audio_file)
                st.audio(audio_file)
                if os.path.exists(audio_file):
                    os.remove(audio_file)

if __name__ == "__main__":
    main()