"""
Scrapes obama speechs from YouTube videos given in obama_addresses.txt
"""
import os
import ffmpeg
from pytube import YouTube

output_dir = 'obama_addresses'

def scrape_video(url):
    """ Given url, download video and audio components """
    content = YouTube(sample_url)
    content_id = url[url.find('=') + 1:]
    audio_name = f'{content_id}_audio'
    video_name = f'{content_id}_video'

    audio = content.streams.filter(type='audio', file_extension='mp4', abr="128kbps")[0]
    video = content.streams.filter(type='video', file_extension='mp4', resolution='1080p')[0]
    audio.download(output_path=output_dir, filename=audio_name)
    video.download(output_path=output_dir, filename=video_name)

    # merge audio video into one
    audio_path = f'{output_dir}/{audio_name}.mp4'
    video_path = f'{output_dir}/{video_name}.mp4'
    audio = ffmpeg.input(audio_path)
    video = ffmpeg.input(video_path)
    ffmpeg.output(audio, video, f'{output_dir}/{content_id}.mp4').run()

    # clean output dir
    os.remove(audio_path)
    os.remove(video_path)

sample_url = 'https://www.youtube.com/watch?v=2AFpAATHXtc'

scrape_video(sample_url)
