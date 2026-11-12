# VideoLingo Q 2.0 — MCP (Model Context Protocol) 接入与使用指南

VideoLingo Q 2.0 现已原生支持 **MCP (Model Context Protocol)**，允许 **ChatGPT、Claude Desktop、Antigravity、Cursor、OpenAI Agents SDK** 等智能体直接控制视频下载、WhisperX 语音识别、三步法字幕翻译、说话人音色配置、多角色 TTS 配音以及批量队列管理。

---

## 一、架构亮点

1. **双传输模式（Stdio + Streamable HTTP / SSE）**
   - **本地 Stdio 模式**：通过 [`start_mcp_server.sh`](file:///Users/niuzhidao/Documents/Program/videolingo/VideoLingo/start_mcp_server.sh) 或 [`backend/mcp_server.py`](file:///Users/niuzhidao/Documents/Program/videolingo/VideoLingo/backend/mcp_server.py) 启动。若检测到 `127.0.0.1:8502` 后端尚未启动，会自动在后台拉起 Vue Studio 后端服务。
   - **HTTP / SSE 网页端同步模式**：当运行 [`start_vue_studio.sh`](file:///Users/niuzhidao/Documents/Program/videolingo/VideoLingo/start_vue_studio.sh) 时，`8502` 端口原生挂载了 MCP 路由，智能体发起的任何操作都会在您的 `http://127.0.0.1:8502` 网页工作台中实时同步显示进度条与画面。
     - **Streamable HTTP 地址**：`http://127.0.0.1:8502/mcp`
     - **SSE 地址**：`http://127.0.0.1:8502/mcp/sse`
     - **MCP 状态与工具自检页**：`http://127.0.0.1:8502/mcp/info`

2. **防超时异步机制（`wait_seconds`）**
   - 视频下载、语音识别与配音耗时较长，所有启动类工具均采用**后台线程异步启动 + 可选短时等待（`wait_seconds`，0~55秒）**，既不会触发 ChatGPT / Claude 客户端的超时中断，又能实时返回进度百分比、当前子步骤、ETA 和最新控制台日志。

---

## 二、客户端接入配置

### 方式 1：本地 Stdio 接入（推荐用于 Antigravity / Claude Desktop / Cursor / ChatGPT 桌面端）

将以下配置添加到您客户端的 `mcpServers` 配置文件中（参考 [`mcp_config.example.json`](file:///Users/niuzhidao/Documents/Program/videolingo/VideoLingo/mcp_config.example.json)）：

```json
{
  "mcpServers": {
    "videolingo-q": {
      "command": "/Users/niuzhidao/Documents/Program/videolingo/VideoLingo/start_mcp_server.sh",
      "args": [],
      "env": {
        "VIDEOLINGO_API_URL": "http://127.0.0.1:8502"
      }
    }
  }
}
```

### 方式 2：HTTP / SSE 接入（适用于支持 URL 连接的客户端或 OpenAI Agents SDK）

先启动 Studio 服务：
```bash
./start_vue_studio.sh
```
然后在客户端中填写：
- **Streamable HTTP URL**：`http://127.0.0.1:8502/mcp`
- **SSE URL**：`http://127.0.0.1:8502/mcp/sse`

---

## 三、全量 MCP 工具列表（共 18 个工具）

### 1. 一键全自动与状态监控
| 工具名称 | 功能说明 |
| :--- | :--- |
| `videolingo_one_shot_process` | **一句话端到端处理**：传入在线视频 `url` 或本地文件路径 `local_file_path`，全自动完成下载/导入、WhisperX 转录、三步翻译、字幕压制与多角色配音合成（支持 `dub_video` / `sub_video` / `sub_only` 模式） |
| `videolingo_get_status` | 查询单视频工作台与批量队列实时状态（百分比、子步骤、ETA、最新日志、生成文件路径），支持 `wait_seconds` 等待任务推进 |
| `videolingo_stop_task` | 立即停止正在运行的单视频或批量任务，或清除错误状态 |

### 2. 配置、音色与大模型管理
| 工具名称 | 功能说明 |
| :--- | :--- |
| `videolingo_get_config` | 读取 `config.yaml` 全部配置、已保存的 LLM 预设、自定义术语及豆包/Edge-TTS 可用音色列表 |
| `videolingo_update_config` | 修改源语言、目标语言、TTS 引擎、男女声音色、是否自动烧录字幕、配音前暂停等配置 |
| `videolingo_manage_llm` | 测试当前大模型 API 连通性与延迟，或加载/保存/删除 LLM 预设（Profile） |

### 3. 单视频分步精细控制（Stage A / B / C）
| 工具名称 | 功能说明 |
| :--- | :--- |
| `videolingo_load_media` | **阶段 A**：下载在线视频/官方字幕，或通过本地文件绝对路径直接导入音视频（`local_video_path`）与外挂字幕（`local_subtitle_path`） |
| `videolingo_run_subtitle_pipeline` | **阶段 B**：启动字幕识别、分句、三步翻译与视频压制流水线（支持 `burn_only` 仅重新压制视频或 `reset_first` 重译） |
| `videolingo_run_dubbing_pipeline` | **阶段 C**：启动说话人性别识别、配音分块与多角色 TTS 配音混流（支持 `force_continue` 确认后继续） |
| `videolingo_workspace_cleanup` | 将当前作品归档到 `history`（`archive`）、清空工作区（`wipe`）、重置字幕缓存或删除配音产物 |

### 4. 深度内容协同（字幕校对、术语修订、说话人微调）
| 工具名称 | 功能说明 |
| :--- | :--- |
| `videolingo_read_subtitles` | 分页或按关键词搜索读取已生成的 SRT 字幕（支持 `src`、`trans`、`src_trans`、`trans_src`、`dub` 及指定批量任务 `task_id`） |
| `videolingo_edit_subtitles` | 按字幕序号精确修改一条或多条字幕文本，并可选 `reburn_video=True` 立即重新烧录视频 |
| `videolingo_get_terminology` | 读取 AI 提取的视频主题摘要（`theme`）、专业术语表（`terms`）、可疑未译词与全局 `domain_vocab` |
| `videolingo_update_terminology` | 新增或修改专业术语翻译、更新视频主题或 `domain_vocab`，并可选 `reapply_to_chunks=True` 重新应用 |
| `videolingo_get_speakers` | 获取识别出的各说话人角色信息、样音台词、可用音色目录及逐句配音台词表 |
| `videolingo_update_speakers` | 修改说话人角色性别与音色，或按句号修改特定台词文本与说话人，并支持 `incremental_rebuild=True` 增量重配音 |

### 5. 批量任务队列管理（Batch Queue）
| 工具名称 | 功能说明 |
| :--- | :--- |
| `videolingo_batch_add_tasks` | 批量添加多个在线视频链接或本地文件路径到队列，并支持 `auto_start=True` 自动开始队列处理 |
| `videolingo_batch_control` | 查询批量队列状态（`state`），或执行启动（`start`）、暂停（`pause`）、停止（`stop`）、清理（`clear_completed`）、单任务运行（`run_single`）与加载到工作台（`load_to_studio`） |

---

## 四、对智能体的常用自然语言指令示例

- **一句话全自动处理**：
  > “帮我用 VideoLingo 把这个视频 `https://www.youtube.com/watch?v=...` 翻译成简体中文并生成多角色配音视频，每隔 15 秒帮我看看进度。”
- **直接处理本地视频**：
  > “把本地视频 `/Users/niuzhidao/Downloads/demo.mp4` 导入 VideoLingo，只生成中英双语字幕视频（不需要配音）。”
- **读取与校对字幕/术语**：
  > “读取当前视频在 VideoLingo 里生成的术语表和双语字幕，看看有没有专业术语翻译不准确的地方，帮我修正并重新压制字幕视频。”
- **调整说话人配音音色**：
  > “看看当前视频识别出了几个说话人？把 SPEAKER_00 设为男声，SPEAKER_01 设为女声，然后重新合成配音。”

## 只读配音额度探查

新增 `videolingo_get_tts_quota`，参数 `provider` 可为 `all`（默认）、`elevenlabs`、`doubao`、`edge`。

MCP `tools/call` 示例：
```json
{"name":"videolingo_get_tts_quota","arguments":{"provider":"all"}}
```

- ElevenLabs：使用已有配音 API Key，请求官方 `GET /v1/user/subscription`，返回已用、总量、剩余、超额、重置 Unix 时间和当前超额额度上限。它是账号/工作区订阅额度，不是单个模型或密钥额度；不换算现金、字数倍率或音频时长。`balance=null` 表示该接口不提供现金余额，不能解读为余额为零。
- 若现有 Key 无权限，返回 `permission_denied`，需要用户单独授权 User Read；本工具绝不自动扩权。认证失败、限流、异常和未配置均有独立状态，不自动重试。
- 自定义 ElevenLabs Base URL 不会收到密钥，返回 `unsupported_endpoint`。只允许官方 HTTPS 地址，禁止跟随重定向。
- 豆包：当前配音 AppID/Access Token 不等同于火山账务 OpenAPI 凭据。官方另有 `ResourcePacksStatus`（资源包余量）和 `QueryBalanceAcct`（账户余额），本版本明确返回 `requires_separate_billing_credentials`，尚未实现这两个适配器；不会寻找或读取额外 AK/SK，不会猜测余额。如需接入，须另行授权最小只读凭据和资源范围。
- Edge：返回 `not_applicable`，没有账户余额接口，不意味着无限额度。
- 每次实时查询，返回 UTC 查询时间；并发任务可能继续消耗额度。本工具不改配置、不调用合成、不启动后端、不写文件、不充值、不改变权限、不改变现有预算保护。网络超时为连接 5 秒、读取 12 秒。

部署：现有运行中进程不会自动加载代码。不要为了这项功能重启正在执行视频任务的服务。独立新 stdio 进程可立即使用新增工具；既有 stdio 客户端在安全时机重新连接后刷新工具列表，HTTP/SSE 服务待视频任务结束后常规重启才能加载。未自动部署/重启。

离线测试（不消耗配音额度）：
```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_tts_quota.py' -v
```

官方依据：
- https://elevenlabs.io/docs/api-reference/user/subscription/get
- https://api.volcengine.com/api-docs/view?action=ResourcePacksStatus&serviceCode=speech_saas_prod&version=2025-05-20
- https://docs.volcengine.com/docs/billing-center/query-balance-acct-query-user-account-balance-information?lang=zh
