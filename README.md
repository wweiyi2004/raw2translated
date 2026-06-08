# raw2translated

`raw2translated` 是一个本地优先(local-first)的流水线工具,用于把无字幕的日语动画转换成结构化转录和字幕文件。

第一个里程碑刻意做得很窄:

- 用 `ffmpeg` / `ffprobe` 抽取并检查媒体
- 产出稳定的转录 JSON 格式
- 导出 `ASS` / `SRT` 字幕
- 把 ASR、说话人分离、角色声纹、翻译等环节留在可替换的接口之后

它不会用用户的媒体去训练公开模型。项目级别的学习应保持在本地:角色声纹、术语表、翻译记忆、用户修正都属于用户自己的项目目录。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
raw2translated --help
```

检查一个视频:

```powershell
raw2translated probe .\input.mkv
```

只安装 ASR 依赖:

```powershell
pip install -e ".[asr]"
```

创建项目输出目录并抽取分析用音频:

```powershell
raw2translated process .\input.mkv --out .\output
```

处理一个无字幕视频,转录其日语音频:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --lang ja
```

可选:用 Demucs 增强人声以提升 ASR 效果:

```powershell
pip install -e ".[preprocess]"
raw2translated process .\input.mkv --out .\output --preprocess demucs --asr faster-whisper --lang ja
```

这会写出 `media/dialogue.wav` 供 ASR 使用。说话人分离默认仍使用 `media/analysis.wav`,以减少对音色身份的破坏。

安装说话人分离支持:

```powershell
pip install -e ".[diarization]"
```

运行 ASR 并把台词分配到说话人聚类:

```powershell
$env:HF_TOKEN="hf_..."
raw2translated process .\input.mkv --out .\output --asr faster-whisper --diarization pyannote --lang ja
```

如果你用的是 pyannote.audio 3.x 环境,请切换模型:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --diarization pyannote --diarization-model pyannote/speaker-diarization-3.1
```

想在 CPU 上更快地冒烟测试,可以用更小的模型:

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --asr-model small --asr-device cpu --asr-compute-type int8
```

从转录 JSON 导出字幕:

```powershell
raw2translated export-subtitle .\output\transcript.speaker.json --format ass --out .\output\episode.ass
```

## MVP 流水线

```text
视频输入
 -> ffprobe 元数据
 -> ffmpeg 抽取分析音频
 -> ASR provider
 -> 强制对齐 provider
 -> 说话人分离 provider
 -> 角色声纹匹配
 -> 翻译 provider
 -> ASS/SRT 导出
 -> MKV 封装
```

当前代码实现了围绕该流水线的稳定外壳。`faster-whisper` ASR 和 `pyannote.audio` 说话人分离已作为可选的模型 provider 接入。角色匹配、强制对齐和翻译仍在开发中。

## 输出目录结构

```text
output/
  manifest.json
  media/
    analysis.wav
    separation_input.wav
    dialogue.wav
  transcript.raw.json
  transcript.speaker.json
  subtitles/
    episode.ass
    episode.srt
```

## 开发

运行标准库测试套件:

```powershell
python -m unittest discover -s tests
```

阶段路线图见 `docs/PROJECT_PLAN.md`,转录 schema 说明见 `docs/DATA_MODEL.md`。
