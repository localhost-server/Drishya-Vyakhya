"""DO REMEMBER TO TURN ON THE LIVE SERVER"""

import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from audio import *
import os

# Importing audio part
# from playsound import playsound
import pygame

st.set_page_config(
    page_title="Chitra Vyakhya",
    layout="centered",
    
)

# RIGHT CLICK OFF
st.markdown(
    """
    <style>
    body {
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
    }
    div.stFileUploader {
        pointer-events: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Hide Menu Style
# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)


## Import the Google Fonts link
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Merienda:wght@800&display=swap');
</style>   
""",unsafe_allow_html=True)
            
# Set Title
st.markdown(
    """<style>
h1 {
    margin-top: -100px;
    font-family: 'Merienda';
    font-size: 60px;
    text-align: center; color:rgb(0, 255, 119);

}
    </style>
    """,
    unsafe_allow_html=True
)
text = '<h1 style="text-align:center; animation: changeColor 20s infinite; transform-style: preserve-3d; perspective: 100px; transform-origin: 40% 60%;">🖼Drishya👀Vyakhya🔠</h1>'
st.markdown(text, unsafe_allow_html=True)
css = """
@keyframes changeColor {
    0% {color: #00FFFF; text-shadow: 0 0 2px #00FFFF, 0 0 24px #00FFFF}
    20% {color: #FF00FF; text-shadow: 0 0 4px #FF00FF, 0 0 22px #FF00FF}
    40% {color: #FFFF00; text-shadow: 0 0 6px #FFFF00, 0 0 20px #FFFF00}
    60% {color: #FF0000; text-shadow: 0 0 8px #FF0000, 0 0 18px #FF0000}
    80% {color: #00FF00; text-shadow: 0 0 10px #00FF00, 0 0 16px #00FF00}
    100% {color: #00FFFF; text-shadow: 0 0 12px #00FFFF, 0 0 14px #00FFFF}
}
"""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# st.markdown("<h1>Divya👀Drishti🖼</h1>", unsafe_allow_html=True)

# Set Background Image
    # background: url("http://192.168.1.64:8501/assets/bg_img.jpg"); 
st.markdown("""<style> [data-testid="stAppViewContainer"] {
    background:url(https://www.cxoinsightme.com/wp-content/uploads/2020/07/AI_shutterstock_1722492775-scaled.jpg);
    background-size: cover;
    }
    </style>""", unsafe_allow_html=True)


# SET MODE
st.markdown('<h2 style="color:aqua; font-family: Merienda;font-size: 30px; margin-top: -40px;">Select Mode:</h2>', unsafe_allow_html=True)
mode = st.radio("hell",
    ["Camera","File Upload"],
    horizontal = True,
    label_visibility = "collapsed"
)
if mode=="Camera":
    img= st.camera_input("",label_visibility="collapsed")
elif mode=="File Upload":
    # UPLOAD IMAGE FILE
    img = st.file_uploader("", type=["jpg", "jpeg", "png"])


# LOAD PROCESSOR
def load_processor():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    return processor

# LOAD MODEL
def load_model():
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large").to("cuda")
    return model


# Define Function to return decoded output
text = '<h2 style="text-align:center;background-image: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Hello, world!</h2>'
st.markdown(text, unsafe_allow_html=True)


def caption(img):
    inputs = load_processor()(img, return_tensors="pt").to("cuda")
    out = load_model().generate(**inputs)
    return load_processor().decode(out[0], skip_special_tokens=False)

def playaud():
    audio=pygame.mixer
    audio.init()
    audio_file="example.wav"
    audio.music.load(audio_file)
    audio.music.play()
    while audio.music.get_busy():
        continue
    audio.music.stop()
    audio.quit()

if img:
    # Generate the caption for the image
    st.image(img, caption="")
    caption = caption(Image.open(img))
    # Display the uploaded image and generated caption
    st.write("Caption: " + caption[:-5])

    # Running the TTS
    mel_output, mel_length, alignment = tacotron().encode_text(caption[:-5])

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifigan().decode_batch(mel_output)

    # Save the waverform
    if os.path.exists("example.wav"):
        os.remove("example.wav")
    else:
        pass
    torchaudio.save("example.wav",waveforms.squeeze(1),sample_rate=22200)
    # torch.cuda.empty_cache()
    # playaud()

    # Load an audio fileaudio=pygame.mixerimport pygame
    # playsound("example.wav")
    st.audio("example.wav")
