import ffmpeg
output_dir = 'obama_addresses'
content_id = 'xAAmF3H0-ek'

audio_path = f'{output_dir}/{content_id}_audio.wav'
video_path = f'{output_dir}/{content_id}_video.mp4'
audio = ffmpeg.input(audio_path)
video = ffmpeg.input(video_path)
ffmpeg.output(audio, video, f'{output_dir}/{content_id}_combined.mp4').run()
    