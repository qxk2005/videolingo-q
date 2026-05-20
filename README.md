<div align="center">

<img src="/docs/logo.png" alt="VideoLingo Logo" height="140">

# VideoLingo Q：连接世界，每一帧都动听

**全自动视频翻译、本地化与配音工具，打造 Netflix 级字幕体验**

</div>

## 🌟 项目概述

VideoLingo Q 是一款一体化的视频搬运神器，旨在生成高质量的 Netflix 级字幕。它不仅能消除生硬的机器翻译，还能通过大模型优化和高精度配音，跨越语言障碍，让全球优质内容触手可及。

**本项目相比于原版的深度优化：**
- **🚀 全程进度可视化**：在 ASR、翻译、分句、分割所有阶段引入实时进度条，精准显示 `(当前/总数)` 数量，掌控每一秒。
- **🤖 大模型术语纠错**：利用大模型智能识别 ASR 过程中的音近字错误，结合领域词汇表实现精准替换，完美同步时间戳。
- **📊 批处理全局监控**：新增批处理模式顶层进度条，直观查看“整体进度”与“当前视频细节”。
- **🎭 Edge TTS 语音试听**：内置最新的官方音色列表，支持在配置时实时试听预览。
- **💄 极致 UI/UX**：优化侧边栏布局，引入局部刷新机制，操作更顺滑，不丢失页面状态。

## 🎥 核心功能

- **🎙️ 高精度识别**：基于 WhisperX 的单词级时间轴识别，低幻觉，精准对齐。
- **📝 智能分割**：结合 NLP 和 AI 语义理解，自动将字幕分割为最适合阅读的单行格式。
- **📚 术语一致性**：支持自定义 + AI 提取术语，确保全篇专业名词翻译统一。
- **🔄 三步翻译法**：直译-反思-意译，三轮迭代，打磨出影视级的地道翻译。
- **🗣️ 一键配音**：支持 GPT-SoVITS, Azure, OpenAI, Edge-TTS 等多种顶级 TTS 方案。
- **📦 批处理模式**：一键处理整个文件夹的视频，高效自动化。

## 🚀 快速开始

### 安装要求
- Python 3.10
- FFmpeg (必须安装)
- NVIDIA GPU (推荐，需安装 CUDA 12.6 与 CUDNN 9.3)

### 安装步骤
1. 克隆仓库：
```bash
git clone https://github.com/qxk2005/videolingo-q.git
cd videolingo-q
```

2. 创建并激活环境：
```bash
conda create -n videolingo python=3.10.0 -y
conda activate videolingo
python install.py
```

3. 启动应用：
```bash
streamlit run st.py
```

## 🛠️ 进阶配置

### LLM 配置
VideoLingo Q 支持多种大模型方案：
- **Gemini CLI**：启用后可获得最精简的配置界面，一键调用。
- **OpenAI 兼容接口**：支持 Claude 3.5, GPT-4, DeepSeek V3 等主流模型。
- **配置文件管理**：支持保存多个 LLM 配置模板，一键切换。

### 配音设置
- **Edge TTS 预览**：在侧边栏选择音色后，点击 🔊 即可试听。
- **人声分离增强**：针对背景噪音大的视频，开启后可显著提升 ASR 准确率。

## 📦 批处理模式
将需要处理的视频放入 `batch/input` 文件夹，配置好 `batch/tasks_setting.xlsx`，运行 `batch/utils/batch_processor.py` 或双击 `OneKeyBatch.bat`。您将看到清晰的多层级进度监控：
- **顶层进度**：显示整体视频处理任务。
- **底层进度**：实时显示当前视频的转录、翻译、分割细节。

## 📄 开源协议
本项目基于 Apache 2.0 协议。特别感谢以下开源项目：
[whisperX](https://github.com/m-bain/whisperX), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [Streamlit](https://streamlit.io/)

---
如果您觉得 VideoLingo Q 提升了您的效率，请给一个 ⭐️ 吧！
