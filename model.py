import torch
import librosa
from pathlib import Path
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC



class Model():
    def __init__(self,  model_path) -> None:
        self.audio_tokenzier, self.audio_model = self._get_models(model_path)
        print("Initialization finished")


    def _get_audio_path(self):
        return str(self.video_file_path.with_suffix(".wav"))

    def _get_models(self, model_path):
        audio_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_path)
        audio_model = Wav2Vec2ForCTC.from_pretrained(model_path)
        print("Model initialized")
        return audio_tokenizer, audio_model

    def _extract_save_audio_file_path(self):
        video_clip = VideoFileClip(str(self.video_file_path))
        clip_end = video_clip.duration
        audio_clip_paths = [] 
        for i in range(0, int(clip_end), 10):
            file_path = f"Data/audio_{i}.mp3"
            sub_end = min(i+10, clip_end)
            sub_clip = video_clip.subclip(i, sub_end)
            sub_clip.audio.write_audiofile(file_path)
            audio_clip_paths.append(file_path)

        return audio_clip_paths

    def _get_audio_data(self):
        audio_data = []
        for audio_path in self.audio_data_paths:
            input_audio, _ = librosa.load(audio_path, sr=16000)
            audio_data.append(input_audio)
        return audio_data

    def _initialize_audio_data(self, video_path):
        self.video_file_path = Path(video_path)
        self.audio_data_paths = self._extract_save_audio_file_path()
        self.audio_data = self._get_audio_data()


    def predict(self, video_path):
        self._initialize_audio_data(video_path=video_path)
        output_text = []
        for data in self.audio_data:
            tokenized_input = self.audio_tokenzier(data, return_tensors="pt").input_values
            logits = self.audio_model(tokenized_input).logits
            logits_idx = torch.argmax(logits, dim=-1)
            transcription = self.audio_tokenzier.batch_decode(logits_idx)
            output_text.append(transcription[0])

        return " ".join(output_text).lower().capitalize()

    
# model = Model("Data/test_video.mp4", "pretrained_model") 
# print(model.predict())    