# raw2translated

`raw2translated` 是一个本地优先(local-first)的流水线工具,用于把无字幕的日语动画转换成结构化转录和字幕文件。

完整闭环:

```text
raw video/audio -> transcript -> translated transcript -> ASS/SRT subtitles
```

第一个里程碑刻意做得很窄:

- 用 `ffmpeg` / `ffprobe` 抽取并检查媒体
- 产出稳定的转录 JSON 格式
- 本地优先的翻译闭环(翻译记忆 / 术语表),无需 API key 即可跑通
- 导出 `ASS` / `SRT` 字幕(原文 / 译文 / 双语)
- 把 ASR、说话人分离、角色声纹、真实翻译模型等环节留在可替换的接口之后

### 不做什么

- 不把用户媒体默认上传到任何云端
- 不用用户的动画文件训练公开模型
- 不内置盗版 / 受版权保护的媒体样本
- 不做自动中文配音 / 声音克隆

项目级别的学习应保持在本地:角色声纹、术语表、翻译记忆、用户修正都属于用户自己的项目目录。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
raw2translated --help
```

> 只想运行核心功能(转录占位、翻译、字幕导出)时,`pip install -e .` 就够了;
> `.[dev]` 额外带上测试和 ruff。真实 ASR / 说话人分离见下文的可选依赖。

检查一个视频:

```powershell
raw2translated probe .\input.mkv
```

### 图形界面 (GUI)

不想用命令行?启动桌面 GUI(基于 Python 自带的 Tkinter,无需额外依赖):

```powershell
raw2translated gui
```

GUI 有三个标签页:

- **Process**:选输入视频/输出目录,选 ASR、翻译 provider、目标语言,点按钮跑流水线,日志实时显示。
- **Editor**:把 transcript JSON 载入表格,逐行查看时间/说话人/原文/译文,手动编辑译文并保存回 JSON。
  可勾选 "Only flagged" 只看未翻译/有备注/低置信度的行;选中一行点 "Play selected" 用 ffplay 试听对应片段(需 ffmpeg)。
- **Export**:选格式(ASS/SRT)和文本模式(原文/译文/双语),一键导出字幕;下方还能把字幕封装(mux)进视频(需 ffmpeg)。

GUI 只是现有 CLI/pipeline 的薄封装,逻辑层(`GuiController`)不依赖 Tkinter,可独立测试。

### 批量处理

把一个目录里的所有媒体文件依次跑流水线,每个文件输出到独立子目录(`out/<文件名>/`):

```powershell
raw2translated batch .\episodes --out .\output --asr faster-whisper --translate memory --translation-memory .\configs\translation_memory.example.json
```

单个文件失败不会中断整批;结尾会汇总成功/失败数量。

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

### 翻译 MVP(本地优先)

翻译默认是 local-first 的,无需任何 API key 或网络。两个 provider:

- `memory`:翻译记忆,按原文整句精确匹配本地 JSON 映射。
- `glossary`:术语表,按词替换,作为可测试的占位翻译。

未命中的句子会保留原文并标记 `untranslated`,不会丢失。

用翻译记忆把 `transcript.speaker.json` 翻成 `transcript.translated.json`:

```powershell
raw2translated translate .\output\transcript.speaker.json --out .\output\transcript.translated.json --provider memory --memory .\configs\translation_memory.example.json
```

也可以在 `process` 阶段一并翻译(默认 `--translate none`,不改变旧行为):

```powershell
raw2translated process .\input.mkv --out .\output --asr faster-whisper --translate memory --translation-memory .\configs\translation_memory.example.json --target-lang zh-CN
```

`manifest.json` 会记录翻译 provider、目标语言和译文输出路径。

### 双语字幕导出

`export-subtitle` 支持三种文本模式:`original`(只日文)、`translated`(只中文)、`bilingual`(中文在上、日文在下)。

```powershell
# 只译文
raw2translated export-subtitle .\output\transcript.translated.json --format ass --out .\output\subtitles\episode.zh.ass --text-mode translated

# 双语 ASS
raw2translated export-subtitle .\output\transcript.translated.json --format ass --out .\output\subtitles\episode.bilingual.ass --text-mode bilingual

# 双语 SRT
raw2translated export-subtitle .\output\transcript.translated.json --format srt --out .\output\subtitles\episode.bilingual.srt --text-mode bilingual
```

`--bilingual` 作为 `--text-mode bilingual` 的兼容别名保留。没有译文的句子会回退到原文,不会输出 `None`。

从转录 JSON 导出字幕(只译文/回退原文是默认):

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
  transcript.translated.json   # 仅在启用翻译时生成
  subtitles/
    episode.ass
    episode.srt
```

源码与配置布局:

```text
src/raw2translated/    CLI、pipeline、各 provider 和 Tkinter GUI
configs/               示例术语表 / 翻译记忆 / 角色 / 字幕样式
docs/                  项目计划、数据模型、实现状态
tests/                 不联网的单元测试(mock/fake provider)
```

## 开发

运行标准库测试套件与 ruff:

```powershell
python -m unittest discover -s tests
python -m ruff check src tests
```

阶段路线图见 `docs/PROJECT_PLAN.md`,转录 schema 说明见 `docs/DATA_MODEL.md`,
当前实现进度见 `docs/IMPLEMENTATION_STATUS.md`。

## 常见问题

**需要 API key 才能翻译吗?**
不需要。默认的 `memory` / `glossary` provider 全部在本地运行。真实机器翻译可以
后续作为新的 provider 接到同一个接口之后。

**没装 faster-whisper / pyannote 能跑测试吗?**
可以。单元测试全部使用 mock/fake provider,不依赖网络、GPU、ffmpeg 或 Hugging Face token。
真实模型验证需要安装对应的可选依赖(`.[asr]`、`.[diarization]`、`.[preprocess]`)。

**翻译漏掉的句子去哪了?**
不会丢。命中不到的句子保留原文,并在 `notes` 里标记 `untranslated`,
字幕导出时回退到原文,不会出现 `None`。

**会上传我的视频吗?**
不会。这是 local-first 工具,默认不把任何用户媒体上传到云端。
