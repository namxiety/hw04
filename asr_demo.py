import whisper

def audio_to_text(audio_path: str, model_name: str = "base"):
    """
    使用Whisper进行语音识别
    :param audio_path: 音频文件路径
    :param model_name: 模型名称（base/small/medium/large）
    :return: 识别结果文本
    """
    # 加载模型（首次运行会自动下载）
    model = whisper.load_model(model_name)
    # 识别音频
    result = model.transcribe(audio_path, language="zh")
    return result["text"]

if __name__ == "__main__":
    # 使用任务二导出的配音音频
    audio_file = "test_audio/hw04_voice.mp3"
    text = audio_to_text(audio_file, model_name="base")
    print("识别结果：")
    print(text)
