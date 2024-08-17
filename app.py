import os
import shutil
from gradio_client import Client
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from transformers import MusicgenForConditionalGeneration
import torch
from transformers import AutoProcessor
import scipy
import gradio as gr
import colorama
from pydub import AudioSegment
from colorama import Fore
import subprocess

import re

def clean_string(string):
    # Usando uma expressão regular para encontrar letras, números e pontos
    padrao = r'[^a-zA-Z0-9.]'
    return re.sub(padrao, '', string)

def rename_file(video_path):
    # Essa parte renomeia o arquivo para input.mp4
    uploaded_filename = video_path.split("/")[2]
    new_filename = "input.mp4"
    os.rename(uploaded_filename, new_filename)



def making_dir():
    #pasta com todos os frames do vídeo

    if not os.path.exists("fotopastas"):
        os.makedirs("fotopastas")
    image_files = [file for file in os.listdir() if file.startswith("frames_")]
    for image in image_files:
        shutil.move(image, os.path.join("fotopastas", image))

    # Defina o caminho para a pasta com as fotos
    pasta = '/content/fotopastas'  # Substitua pelo caminho da sua pasta

    # Lista de extensões de arquivos de imagem que você deseja processar
    extensoes_de_imagem = ['.jpg', '.png', '.jpeg']

    # Ordenando os arquivos
    arquivos_ordenados = sorted(
        [arquivo for arquivo in os.listdir(pasta) if any(arquivo.lower().endswith(ext) for ext in extensoes_de_imagem)],
        key=lambda arquivo: int(arquivo.split("_")[1].split(".")[0])
    )

    return [arquivos_ordenados,pasta]

def frame_list(video_path,seconds):

    rename_file(video_path)

    # ffmpeg -i input.mp4 -vf "fps=1/$seconds" -q:v 2 frames_%03d.jpg
    command = [
    'ffmpeg',
    '-i', 'input.mp4',
    '-vf', f'fps=1/{seconds}',
    '-q:v', '2',
    'frames_%03d.jpg'
    ]

# Run the command using subprocess
    subprocess.run(command)
    #pasta com todos os frames do vídeo

    elements = making_dir()

    from gradio_client import Client

    # Inicialize o cliente
    client = Client("https://fffiloni-clip-interrogator-2.hf.space/")

    finalList = []

    # Loop para percorrer as fotos na pasta
    for arquivo in elements[0]:
        caminho_arquivo = os.path.join(elements[1], arquivo)
        result = client.predict(
            caminho_arquivo,
            "best",
            8,
            api_name="/clipi2"
        )
        newList = []
        for item in result:
            if isinstance(item, str) and "{" in item:
                break
            newList.append(item)

        newString = newList[0] if newList else ""
        finalList.append(newString)

    resultList = []

    for description in finalList:
        first = description.split(',')
        resultList.append(first[0])
    print(resultList)
    return resultList


def langchain_handle_text(text):
    print(Fore.CYAN + "to no lang")
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    llm = OpenAI(temperature=0.3,model_name="gpt-3.5-turbo")
    conversation = ConversationChain(

    llm=llm,
    verbose=True,

    memory=ConversationBufferMemory()
    )

    conversation.predict(input=f"Given a text and you being an internationally renowned melodist, create a melody description with instruments and necessary transitions according to the context of the text. The text:{text}")
    output = conversation.predict(input="Summarize the melody without removing the necessary instruments and transitions. the otuput should be : the melody begins...")
    print(output)

    return output


def eleven_labs(prompt):
  import requests

  CHUNK_SIZE = 1024
  url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"

  headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": "ELEVENLABS_API_KEY"
  }

  data = {
    "text": prompt,
    "model_id": "eleven_multilingual_v1",
    "voice_settings": {
      "stability": 0.5,
      "similarity_boost": 0.5
      
    }
  }

  response = requests.post(url, json=data, headers=headers)
  print(response.text)
  with open('narracao.mp3', 'wb') as f:
      for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
          if chunk:
              f.write(chunk)


def check_duration():
  # Carregue o arquivo de áudio
  audio1 = AudioSegment.from_file("audio.mp3", format="mp3")
  audio2 = AudioSegment.from_file("narracao.mp3", format="mp3")

  # Obtenha a duração em milissegundos
  duração_em_milissegundos = len(audio1)
  duração_em_milissegundos2 = len(audio2)

  # Converta a duração para segundos
  duração_em_segundos = duração_em_milissegundos / 1000
  duração_em_segundos2 = duração_em_milissegundos2 / 1000

  print(f"A duração do áudio é de {duração_em_segundos} segundos.")
  print(f"A duração do áudio é de {duração_em_segundos2} segundos.")
  if duração_em_segundos > duração_em_segundos2:
    maior = duração_em_segundos
  else:
    maior = duração_em_segundos2

  return maior


def merge_audio_text():
  
  #ffmpeg -y -i audio_1.wav -vn -ar 44100 -ac 2 -b:a 192k audio.mp3
  subprocess.run(['ffmpeg', '-y', '-i', 'audio_1.wav', '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', 'audio.mp3'])
  duration = check_duration()
  #ffmpeg -stream_loop -1 -i audio.mp3 -t "$duration" -c:a libmp3lame audio_loop.mp3
  subprocess.run(['ffmpeg', '-stream_loop', '-1', '-i', 'audio.mp3', '-t', str(duration), '-c:a', 'libmp3lame', 'audio_loop.mp3'])
  
  #ffmpeg -i narracao.mp3 -i audio_loop.mp3 -filter_complex amix=inputs=2:duration=first:dropout_transition=2 output.mp3
  subprocess.run(['ffmpeg', '-i', 'narracao.mp3', '-i', 'audio_loop.mp3', '-filter_complex', 'amix=inputs=2:duration=first:dropout_transition=2', 'output.mp3'])
  audio_final = '/content/output.mp3'
  return audio_final


def langchain_handle(description):
    print(Fore.CYAN + "to no lang")
    os.environ["OPENAI_API_KEY"] = "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"
    llm = OpenAI(temperature=0.3,model_name="gpt-3.5-turbo")
    conversation = ConversationChain(

    llm=llm,
    verbose=True,

    memory=ConversationBufferMemory()
    )

    conversation.predict(input=f"given a list of phrases and you being a world-renowned melodist, create a melody based on the context generated by the phrases on the list, reporting the necessary instruments and their transitions. The list:{description}")
    conversation.predict(input="put the intro and all the scenes together in one phrase. Give me the output star with: the melody begins ")
    y = conversation.predict(input='Summarize the and starts with: the melody begins')
    print(y)
    return y

def music_gen(description):
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device);
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[f"{description}"],
        padding=True,
        return_tensors="pt",
    )
    print('antes do sampling')
    sampling_rate = model.config.audio_encoder.sampling_rate
    print('depois do sampling')

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=1503)

    # Audio(audio_values[0].cpu().numpy(), rate=sampling_rate)
    print('vou salvar o audio')

    nome = 'audio_1.wav'
    scipy.io.wavfile.write(nome, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

    return "/content/audio_1.wav"


def merge_audio_video():

    # ffmpeg -y -i audio_1.wav -vn -ar 44100 -ac 2 -b:a 192k audio.mp3

    # ffmpeg -y -i input.mp4 -i audio.mp3 -c:v copy -c:a copy output.mp4

    subprocess.run(['ffmpeg', '-y', '-i', 'audio_1.wav', '-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k', 'audio.mp3'])

    # Combinar input.mp4 com audio.mp3 em output.mp4
    subprocess.run(['ffmpeg', '-y', '-i', 'input.mp4', '-i', 'audio.mp3', '-c:v', 'copy', '-c:a', 'copy', 'output.mp4'])


def handle_text(text):
  description = langchain_handle_text(text)
  audio = music_gen(description)
  eleven_labs(text)
  audio_final = merge_audio_text()
  return audio_final

import gradio as gr
from pytube import YouTube
def download_youtube_video(youtube_link,seconds):

        # Create a YouTube object for the provided link
        yt = YouTube(youtube_link)

        # Get the highest resolution stream (You can customize this)
        video_stream = yt.streams.filter(resolution = '720p',only_video=True).first()

        yt.title = clean_string(yt.title)
        # Download the video
        video_stream.download(output_path = '/content', filename = f'{yt.title}.mp4')

        video_path = f"/content/{yt.title}.mp4"
        print(video_path)
        print(yt.length)
        description = frame_list(video_path,seconds)

        final_description = langchain_handle(description)
        audio_path = music_gen(final_description)
        merge_audio_video()
        new_video_path = '/content/output.mp4'
        return new_video_path





iface_1 = gr.Interface(
    download_youtube_video,
    [gr.Textbox(label="Enter YouTube Video Link"),
     gr.Dropdown( ["5", "3", "1"], label="Seconds", info="Extract an image every chosen number of seconds")],
    "video",

)
iface_2 = gr.Interface(
    handle_text,
    gr.Textbox(label="Enter a Text"),
    "audio"
)


# iface_1.launch(share = True,debug=True,enable_queue=True)
demo = gr.TabbedInterface([iface_1, iface_2], ["video-to-SoundClip", "video-to-NarrativeText"])
demo.launch(share=True,debug=True,enable_queue=True)
