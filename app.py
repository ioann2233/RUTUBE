import cv2
import os
import numpy as np
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from pydub import AudioSegment
import whisper
import streamlit as st
import json
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from moviepy.editor import VideoFileClip, concatenate_videoclips

def split_video_into_scenes(video_path, output_dir, threshold=27.0, min_scene_length=15):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    filtered_scenes = []
    
    for scene in scene_list:
        start_time, end_time = scene
        duration = (end_time.get_frames() - start_time.get_frames()) / fps
        if duration >= min_scene_length:
            filtered_scenes.append(scene)

    for i, scene in enumerate(filtered_scenes):
        output_file = os.path.join(output_dir, f"video_{i + 1}.mp4")
        split_video_ffmpeg(video_path, [scene], output_file_template=output_file)

def process_audio(audio_path):
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment = audio_segment.set_channels(1)
    audio_segment = audio_segment.set_frame_rate(16000)
    processed_audio_path = audio_path.replace(".wav", "_processed.wav")
    audio_segment.export(processed_audio_path, format="wav")
    return processed_audio_path

def recognize_audio(audio_path):
    processed_audio_path = process_audio(audio_path)
    model_whisper = whisper.load_model("large")
    result = model_whisper.transcribe(processed_audio_path, language="ru", fp16=False, word_timestamps=True)

    subtitles = []
    segments = result['segments']
    for segment in segments:
        for word_info in segment['words']:
            word = word_info['word']
            start = word_info['start']
            end = word_info['end']
            subtitles.append({"word": word, "start": start, "end": end})

    return subtitles

def save_subtitles(subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)

def create_shorts(output_dir, video_path, num_shorts, short_duration):
    shorts_dir = os.path.join(output_dir, "shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    
    clip = VideoFileClip(video_path)
    total_duration = clip.duration

    for i in range(num_shorts):
        start_time = np.random.uniform(0, total_duration - short_duration)
        end_time = start_time + short_duration
        output_file = os.path.join(shorts_dir, f"short_{i + 1}.mp4")
        clip.subclip(start_time, end_time).write_videofile(output_file, codec="libx264", audio_codec="aac")

st.title("Генерация виральных клипов")

video_file = st.file_uploader("Загрузите видео файл", type=["mp4"])
if video_file is not None:
    video_path = os.path.join(os.getcwd(), video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    output_dir = os.path.join(os.getcwd(), "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    split_video_into_scenes(video_path, output_dir)

    audio_file = st.file_uploader("Загрузите аудиофайл", type=["wav"])
    if audio_file is not None:
        audio_path = os.path.join(os.getcwd(), audio_file.name)
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        subtitles = recognize_audio(audio_path)
        subtitles_path = os.path.join(output_dir, "subtitles.json")
        save_subtitles(subtitles, subtitles_path)

        st.write("Редактируйте субтитры ниже:")
        edited_subtitles = st.text_area("Субтитры", json.dumps(subtitles, ensure_ascii=False, indent=4))
if st.button("Сохранить субтитры"):
            edited_subtitles_dict = json.loads(edited_subtitles)
            save_subtitles(edited_subtitles_dict, subtitles_path)
            st.success("Субтитры успешно сохранены!")

        num_shorts = st.number_input("Количество генерируемых шортсов", min_value=1, max_value=100, value=5)
        short_duration = st.number_input("Длительность каждого шортса (в секундах)", min_value=1, max_value=60, value=30)

        if st.button("Создать итоговое видео"):
            create_shorts(output_dir, video_path, num_shorts, short_duration)
            st.success("Итоговые шортсы успешно созданы!")

        if st.button("Сохранить сгенерированные шортсы"):
            shorts_path = os.path.join(output_dir, "shorts")
            os.makedirs(shorts_path, exist_ok=True)
            save_subtitles(edited_subtitles_dict, os.path.join(shorts_path, "subtitles.json"))
            st.success("Шортсы успешно сохранены в папке 'shorts'!")

model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')

def predict_frame_importance(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return np.mean(features)

def get_scene_frames(video_path, threshold=0.52):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frame_rate = clip.fps
    scene_frames = []

    for start_time in range(5, int(duration) - 5, 5):
        end_time = min(start_time + 5, duration)
        mid_time = (start_time + end_time) / 2

        frames_to_check = [
            (start_time, clip.get_frame(start_time)),
            (mid_time, clip.get_frame(mid_time)),
            (end_time, clip.get_frame(end_time))
        ]

        for time, frame in frames_to_check:
            importance = predict_frame_importance(frame)
            if importance >= threshold:
                print(f"Кадр {time}: вероятность попадания в трейлер — {importance}")
                scene_frames.append(time)
                break

    return scene_frames, duration, frame_rate

def create_cuts(video_path, key_frames, duration, frame_rate):
    cut_videos = []
    seen_frames = set()
    last_end_time = -1

    start_limit = 5
    end_limit = duration - 1

    for frame_idx in key_frames:
        start_time = max(start_limit, (frame_idx / frame_rate) - 20)
        end_time = min(end_limit, (frame_idx / frame_rate) + 20)
        if (start_time, end_time) not in seen_frames and start_time >= last_end_time:
            output_file = f"cut_video_{frame_idx}.mp4"
            print(f"Вырезаем: {output_file} с {start_time:.2f} до {end_time:.2f}")

            try:
                clip = VideoFileClip(video_path).subclip(start_time, end_time)
                clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
                cut_videos.append(output_file)
                seen_frames.add((start_time, end_time))
                last_end_time = end_time
            except Exception as e:
                print(f"Ошибка при обработке видео: {e}")

    return cut_videos

def create_final_highlight(cut_videos, output_file):
    if cut_videos:
        clips = [VideoFileClip(video) for video in cut_videos]
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")
        print(f"Финальное видео создано: {output_file}")
    else:
        print("Нет доступных клипов для создания финального видео.")

if __name__ == "__main__":
    st.write("Загрузка и анализ видео...")

    key_frames, duration, frame_rate = get_scene_frames(video_path)
    cut_videos = create_cuts(video_path, key_frames, duration, frame_rate)
    create_final_highlight(cut_videos, "final_highlight.mp4")
