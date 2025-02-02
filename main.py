import time
import speech_recognition as sr
from groq import Groq
from PIL import ImageGrab, Image
from dotenv import load_dotenv
import google.generativeai as genai
import os
import streamlit as st
import base64
import asyncio  
import edge_tts
import pygame  # Add this import for audio playback

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Initialize pygame mixer
pygame.mixer.init()

load_dotenv()

# API Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
genai_api_key = os.getenv("GENAI_API_KEY")

groq_client = Groq(api_key=groq_api_key)
genai.configure(api_key=genai_api_key)

# Model Configuration
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'}
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                            generation_config=generation_config, 
                            safety_settings=safety_settings)

# System message for the AI
sys_msg = (
    'You are a multi-modal AI voice assistant named Prometheus. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity. also call me Master while replying'
)

convo = [
    {'role': 'system', 'content': sys_msg}
]

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Could not request results"

async def speak_async(response, voice='en-US-MichelleNeural', output_file='temp_output.mp3'):
    try:
        # Generate the audio file
        communicate = edge_tts.Communicate(response, voice)
        await communicate.save(output_file)
        
        # Play using pygame
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"Error in speak_async: {e}")
    finally:
        # Clean up the temporary file after playing
        try:
            pygame.mixer.music.unload()
            os.remove(output_file)
        except:
            pass

def speak(response, voice='en-US-MichelleNeural'):
    asyncio.run(speak_async(response, voice))

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(
        messages=convo, 
        model='llama3-70b-8192'
    )
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model named AVA. You will determine whether extracting the users clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the users prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["take screenshot", "None"]. '
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )
    convo = [
        {'role': 'system', 'content': sys_msg},
        {'role': 'user', 'content': prompt}
    ]
    chat_completion = groq_client.chat.completions.create(
        messages=convo, 
        model='llama3-70b-8192'
    )
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI '
        'assistant to the user. Instead take the user prompt input and try to extract all meaning '
        'from the photo relevant to the user prompt. Then generate as much objective data about '
        'the image for the AI assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def start_listening():
    while True:
        user_input = get_voice_input()
        if user_input and user_input.lower() != "could not understand audio":
            st.write(f"**User said:** {user_input}")
            
            # Process the input
            call = function_call(user_input)
            if 'take screenshot' in call:
                print('Taking screenshot')
                take_screenshot()
                visual_context = vision_prompt(prompt=user_input, photo_path='screenshot.jpg')
            else:
                visual_context = None

            response = groq_prompt(prompt=user_input, img_context=visual_context)
            
            # Display and speak the response
            st.write(f"**Assistant Response:** {response}")
            speak(response)
            
            with open('report.txt', 'a') as file:
                file.write(f"User: {user_input}\n")
                file.write(f"AI: {response}\n")
                file.write("-" * 40 + "\n")

        time.sleep(0.1)

# UI Configuration
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
    }

    .stButton button {
        width: 150px;
        height: 150px;
        font-size: 20px;
        margin-top: 200px;
        margin-left: auto;
        margin-right: auto;
        display: block;
        border-radius: 50%;
    }

    .loader {
        border: 16px solid #f3f3f3;
        border-radius: 50%;
        border-top: 16px solid #3498db;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .listening-text {
        margin-top: 10px;
        font-size: 24px;
        color: #3498db;
        margin-left:20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
image_path = "prometheus.png"
image_base64 = get_base64_of_bin_file(image_path)
st.sidebar.markdown(
    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
    f'<h2 style="font-family: Montserrat; font-size: 30px;">Prometheus.ai</h2>'
    f'<img src="data:image/png;base64,{image_base64}" style="width: 100px; margin-left: 2px;"></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("Multi Model Voice Assistant")

def start_button_clicked():
    st.session_state.button_clicked = True

if st.session_state.button_clicked:
    st.markdown(
        '''
        <div class="centered">
            <div class="loader"></div>
            <div class="listening-text">Listening...</div>
        </div>
        ''',
        unsafe_allow_html=True
    )
    start_listening()
else:
    st.button("START", on_click=start_button_clicked)