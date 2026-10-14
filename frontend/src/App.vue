<template>
  <div class="min-h-screen flex flex-col">
    <!-- ==================== 1. 极简顶部导航栏 (Simplified Top Header) ==================== -->
    <header class="studio-card sticky top-0 z-30 px-5 py-2.5 flex items-center justify-between shadow-sm">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-sm shrink-0">
          VL
        </div>
        <span class="font-bold text-base tracking-tight whitespace-nowrap">VideoLingo Q</span>
        <span v-if="wsState.task?.running" class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-amber-500/15 text-amber-500 border border-amber-500/30 animate-pulse">
          ⏳ {{ wsState.task.step_name || '后台任务执行中...' }}
        </span>
      </div>

      <!-- Center: Mode Switcher -->
      <div class="studio-subtle p-1 rounded-xl flex items-center gap-1">
        <button @click="workspaceMode = 'single'" :class="workspaceMode === 'single' ? 'bg-blue-600 text-white shadow-sm font-semibold' : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 font-medium'" class="px-3.5 py-1.5 rounded-lg text-xs transition whitespace-nowrap">
          🎬 单视频工作台
        </button>
        <button @click="switchBatchMode()" :class="workspaceMode === 'batch' ? 'bg-blue-600 text-white shadow-sm font-semibold' : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 font-medium'" class="px-3.5 py-1.5 rounded-lg text-xs transition flex items-center gap-1.5 whitespace-nowrap">
          📦 批量队列
          <span v-if="batchTasks.length" class="px-1.5 py-0.2 rounded-full bg-amber-500/20 text-amber-500 text-xs font-bold">{{ batchTasks.length }}</span>
        </button>
      </div>

      <!-- Right: Concise Action Controls -->
      <div class="flex items-center gap-2.5">
        <button @click="openSettings('llm')" class="studio-subtle px-3.5 py-1.5 rounded-lg text-xs font-semibold hover:border-blue-500 transition flex items-center gap-1.5 whitespace-nowrap">
          <span>⚙️</span>
          <span>参数设置</span>
          <span class="text-slate-400 font-normal hidden sm:inline">(LLM / 字幕 / 配音)</span>
        </button>

        <button @click="toggleTheme" class="studio-subtle px-3 py-1.5 rounded-lg text-xs font-medium hover:border-blue-500 transition flex items-center gap-1.5 whitespace-nowrap">
          <span>{{ isDark ? '☀️' : '🌙' }}</span>
          <span>{{ isDark ? '浅色模式' : '深色模式' }}</span>
        </button>

        <button v-if="wsState.task?.running" @click="stopTask" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500 text-white hover:bg-rose-600 transition whitespace-nowrap">
          ⏹ 停止任务
        </button>
      </div>
    </header>

    <!-- ==================== 2. 主体双栏工作区 (2-Column Studio: 9 + 3) ==================== -->
    <div v-if="workspaceMode === 'single'" class="flex-1 grid grid-cols-12 gap-4 p-4 max-w-[1800px] w-full mx-auto">

      <!-- ==================== 左主栏 (9 Cols / 75% Width) ==================== -->
      <main class="col-span-12 lg:col-span-9 flex flex-col gap-4">

        <!-- 全局错误横幅 (如有) -->
        <div v-if="wsState.task?.error" class="studio-card rounded-xl p-4 border-l-4 border-rose-500 bg-rose-500/5 flex flex-col gap-2">
          <div class="flex items-center justify-between">
            <span class="font-bold text-rose-500 text-sm">❌ 任务执行遇到异常</span>
            <button @click="clearError" class="px-3 py-1 rounded-lg bg-rose-500 text-white text-xs font-semibold">清除错误并重试</button>
          </div>
          <pre class="text-xs font-mono p-2.5 rounded bg-slate-950 text-rose-300 overflow-x-auto whitespace-pre-wrap max-h-40">{{ wsState.task.error }}</pre>
        </div>

        <!-- 2A. 上方统一多源视频监视器 -->
        <section class="studio-card rounded-xl overflow-hidden shadow-sm">
          <div class="px-4 py-2.5 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-2">
            <div class="flex items-center gap-1.5">
              <span class="text-xs font-bold text-slate-400 mr-1">🎬 信号源:</span>
              <button @click="selectVideoSource('orig')" :class="activeVideoSource === 'orig' ? 'bg-blue-600 text-white font-semibold shadow-sm' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-1 rounded-md text-xs transition">
                原始视频
              </button>
              <button v-if="wsState.video?.vtt_url" @click="selectVideoSource('vtt')" :class="activeVideoSource === 'vtt' ? 'bg-blue-600 text-white font-semibold shadow-sm' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-1 rounded-md text-xs transition">
                外挂官方字幕 (VTT)
              </button>
              <button v-if="wsState.video?.sub_url" @click="selectVideoSource('sub')" :class="activeVideoSource === 'sub' ? 'bg-blue-600 text-white font-semibold shadow-sm' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-1 rounded-md text-xs transition">
                烧录双语成片 (Sub)
              </button>
              <button v-if="wsState.video?.dub_url" @click="selectVideoSource('dub')" :class="activeVideoSource === 'dub' ? 'bg-blue-600 text-white font-semibold shadow-sm' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-1 rounded-md text-xs transition">
                🎙️ 多角色配音成片 (Dub)
              </button>
            </div>
            <div class="flex items-center gap-2 text-xs">
              <span v-if="activeLineInfo" class="px-2.5 py-0.5 rounded-full bg-blue-500/15 text-blue-500 font-semibold">
                {{ activeLineInfo }}
              </span>
              <span class="font-mono text-slate-400">{{ wsState.video?.file_name || '未加载视频' }}</span>
            </div>
          </div>

          <div class="relative bg-slate-950 flex items-center justify-center min-h-[260px] max-h-[420px]">
            <video
              v-if="currentVideoUrl"
              ref="videoPlayerRef"
              :key="currentVideoUrl + '_' + activeVideoSource"
              :src="currentVideoUrl"
              controls
              crossorigin="anonymous"
              class="w-full max-h-[420px] object-contain"
            >
              <track
                v-if="activeVideoSource === 'vtt' && wsState.video?.vtt_url"
                kind="subtitles"
                label="Official Subtitle"
                srclang="en"
                :src="wsState.video.vtt_url"
                default
              />
            </video>
            <div v-else class="py-16 text-center text-slate-400 flex flex-col items-center gap-2">
              <span class="text-3xl">🎬</span>
              <span class="text-sm font-semibold">当前暂无视频文件</span>
              <span class="text-xs text-slate-500">请在下方「阶段 A · 片源获取」下载 YouTube 视频或拖拽上传本地音视频</span>
            </div>
          </div>
        </section>

        <!-- 2B. 下方三阶段步进工作台 -->
        <section class="studio-card rounded-xl p-4 flex flex-col gap-4 shadow-sm flex-1">
          <div class="grid grid-cols-3 gap-3">
            <button @click="activeStage = 'a'" :class="activeStage === 'a' ? 'border-2 border-blue-600 bg-blue-500/10' : 'studio-subtle hover:border-blue-500'" class="p-3 rounded-xl text-left transition flex items-center justify-between">
              <div>
                <div class="text-xs text-slate-400 font-medium">阶段 A · 片源获取</div>
                <div class="text-sm font-bold mt-0.5">下载视频 / 提取字幕</div>
              </div>
              <span v-if="wsState.task?.running && wsState.task?.stage === 'download'" class="px-2 py-0.5 rounded bg-blue-600 text-white text-xs font-bold animate-pulse">⏳ {{ (wsState.task.percent || 0).toFixed(0) }}%</span>
              <span v-else class="text-sm font-bold">{{ wsState.video?.exists ? '✅' : '1' }}</span>
            </button>

            <button @click="activeStage = 'b'" :class="activeStage === 'b' ? 'border-2 border-blue-600 bg-blue-500/10' : 'studio-subtle hover:border-blue-500'" class="p-3 rounded-xl text-left transition flex items-center justify-between">
              <div>
                <div class="text-xs text-slate-400 font-medium">阶段 B · 字幕与翻译</div>
                <div class="text-sm font-bold mt-0.5">三步翻译与术语纠错</div>
              </div>
              <span v-if="wsState.task?.running && wsState.task?.stage === 'subtitle'" class="px-2 py-0.5 rounded bg-blue-600 text-white text-xs font-bold animate-pulse">⏳ {{ (wsState.task.percent || 0).toFixed(0) }}%</span>
              <span v-else class="text-sm font-bold">{{ wsState.stage_b?.src_srt_exists ? '✅' : '2' }}</span>
            </button>

            <button @click="activeStage = 'c'" :class="activeStage === 'c' ? 'border-2 border-blue-600 bg-blue-500/10' : 'studio-subtle hover:border-blue-500'" class="p-3 rounded-xl text-left transition flex items-center justify-between">
              <div>
                <div class="text-xs text-slate-400 font-medium">阶段 C · 多角色配音</div>
                <div class="text-sm font-bold mt-0.5">说话人微调与合成</div>
              </div>
              <span v-if="wsState.task?.running && wsState.task?.stage === 'dubbing'" class="px-2 py-0.5 rounded bg-blue-600 text-white text-xs font-bold animate-pulse">⏳ {{ (wsState.task.percent || 0).toFixed(0) }}%</span>
              <span v-else-if="wsState.stage_c?.waiting_confirm" class="px-2 py-0.5 rounded bg-amber-500 text-white text-xs font-bold whitespace-nowrap">⏸️ 待确认</span>
              <span v-else class="text-sm font-bold">{{ wsState.stage_c?.dub_video_exists ? '✅' : '3' }}</span>
            </button>
          </div>

          <!-- ==================== 实时任务进度监控卡片 (所有阶段共用内嵌反馈) ==================== -->
          <div v-if="wsState.task?.running" class="rounded-xl p-4 border-2 border-blue-500/40 bg-blue-500/5 flex flex-col gap-3 shadow-sm">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex items-center gap-2.5">
                <span class="w-3 h-3 rounded-full bg-blue-600 animate-ping"></span>
                <div>
                  <div class="text-sm font-bold text-blue-600 dark:text-blue-400 flex items-center gap-2">
                    <span>{{ wsState.task.step_name || '正在执行任务...' }}</span>
                    <span v-if="wsState.task.elapsed_sec" class="px-2 py-0.5 rounded bg-slate-500/10 text-slate-500 dark:text-slate-400 font-mono text-xs font-normal">
                      ⏱️ 已用时 {{ formatElapsed(wsState.task.elapsed_sec) }}
                    </span>
                  </div>
                  <div class="text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono">
                    {{ wsState.task.sub_step || '正在处理中，请稍候...' }}
                  </div>
                </div>
              </div>

              <div class="flex flex-wrap items-center gap-2 text-xs font-mono">
                <span v-if="wsState.task.total_size" class="px-2.5 py-1 rounded-lg studio-card font-semibold">
                  📦 大小: {{ wsState.task.total_size }}
                </span>
                <span v-if="wsState.task.speed" class="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 font-bold">
                  🚀 速度: {{ wsState.task.speed }}
                </span>
                <span v-if="wsState.task.eta" class="px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20 font-bold">
                  ⏳ 剩余: {{ wsState.task.eta }}
                </span>
                <span class="px-3 py-1 rounded-lg bg-blue-600 text-white font-bold text-sm">
                  {{ Number(wsState.task.percent || 0).toFixed(1) }}%
                </span>
              </div>
            </div>

            <div class="w-full h-3 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden p-0.5">
              <div
                class="h-full rounded-full bg-gradient-to-r from-blue-600 via-indigo-500 to-emerald-500 transition-all duration-300"
                :style="{ width: Math.max(4, Number(wsState.task.percent || 4)) + '%' }"
              ></div>
            </div>

            <div v-if="wsState.task.stage === 'download'" class="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div
                v-for="(label, idx) in ['1. 解析链接与 Cookie', '2. 拉取封面与音视频流', '3. FFmpeg 音画合并', '4. 探测与清洗官方字幕']"
                :key="idx"
                class="px-2.5 py-1.5 rounded-lg flex items-center justify-between border"
                :class="wsState.task.step_index > idx + 1 ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-600 dark:text-emerald-400 font-semibold' : (wsState.task.step_index === idx + 1 ? 'bg-blue-600 text-white border-blue-600 font-bold' : 'studio-card text-slate-400 border-transparent')"
              >
                <span class="truncate">{{ label }}</span>
                <span>{{ wsState.task.step_index > idx + 1 ? '✅' : (wsState.task.step_index === idx + 1 ? '⏳' : '⚪') }}</span>
              </div>
            </div>

            <div class="rounded-lg bg-slate-950 text-slate-300 p-2.5 font-mono text-xs flex flex-col gap-1 border border-slate-800">
              <div class="flex items-center justify-between text-slate-400 pb-1 border-b border-slate-800/80">
                <span class="flex items-center gap-1.5 text-emerald-400 font-semibold">
                  <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  实时执行日志流 (最近 5 条)
                </span>
                <button @click="showConsole = !showConsole" class="text-blue-400 hover:underline">
                  {{ showConsole ? '收起底部完整终端' : '展开底部完整终端 ▲' }}
                </button>
              </div>
              <div v-if="!wsState.logs?.length" class="text-slate-500 py-1">正在等待后端输出日志...</div>
              <div v-for="(line, i) in (wsState.logs || []).slice(-5)" :key="i" class="truncate leading-relaxed text-slate-200">
                {{ line }}
              </div>
            </div>
          </div>

          <!-- STAGE A -->
          <div v-if="activeStage === 'a'" class="flex flex-col gap-3.5 text-xs">
            <div class="studio-subtle p-4 rounded-xl flex flex-col gap-3">
              <div class="flex items-center justify-between">
                <span class="font-bold text-sm">🔗 YouTube 视频链接与极速字幕通道</span>
                <button @click="openSettings('sub')" class="text-blue-500 hover:underline font-medium">⚙️ 配置 Cookies 与防风控</button>
              </div>
              <div class="flex gap-2">
                <input v-model="ytbUrl" type="text" placeholder="粘贴 YouTube 链接 (如 https://www.youtube.com/watch?v=...)" class="studio-input flex-1 px-3 py-2 rounded-lg font-mono text-xs" />
                <select v-model="ytbRes" class="studio-input px-3 py-2 rounded-lg font-semibold">
                  <option value="360">360p 极速</option>
                  <option value="1080">1080p 高清</option>
                  <option value="best">Best 最佳画质</option>
                </select>
              </div>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                <button @click="downloadYoutube(false)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-blue-600 text-white font-semibold hover:bg-blue-700 disabled:opacity-50 transition">
                  {{ wsState.task?.running && wsState.task?.stage === 'download' ? `⏳ 正在下载 (${(wsState.task.percent || 0).toFixed(0)}%)...` : '📥 下载视频 (自动探测官方字幕)' }}
                </button>
                <button @click="downloadYoutube(true)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-indigo-600/15 text-indigo-600 dark:text-indigo-400 border border-indigo-500/30 font-semibold hover:bg-indigo-600 hover:text-white disabled:opacity-50 transition">
                  ⚡ 仅下载官方字幕 (秒级跳过转录)
                </button>
                <button v-if="wsState.video?.exists || wsState.files?.length" @click="deleteVideo" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20 font-semibold hover:bg-rose-500 hover:text-white disabled:opacity-50 transition">
                  🗑️ 删除当前视频并清空工作区
                </button>
              </div>
            </div>

            <label class="studio-subtle border-2 border-dashed border-slate-300 dark:border-slate-700 rounded-xl p-6 text-center cursor-pointer hover:border-blue-500 transition block">
              <input type="file" class="hidden" accept=".mp4,.mov,.mkv,.avi,.flv,.wmv,.webm,.mp3,.wav,.m4a,.flac" @change="uploadVideoFile" />
              <div class="text-base font-semibold mb-1">📤 点击或拖拽本地视频 / 音频文件上传</div>
              <div class="text-slate-400">支持 MP4, MOV, MKV, AVI, WEBM, MP3, WAV, M4A（音频将自动合成黑屏视频流）</div>
            </label>
          </div>

          <!-- STAGE B -->
          <div v-if="activeStage === 'b'" class="flex flex-col gap-3.5 text-xs">
            <div class="studio-subtle p-3.5 rounded-xl flex flex-col gap-2.5">
              <div class="flex items-center justify-between">
                <span class="font-bold text-sm">🚀 字幕与三步翻译流水线 (6 步)</span>
                <button @click="openSettings('sub')" class="text-blue-500 hover:underline font-medium">⚙️ 打开字幕与领域词表设置</button>
              </div>
              <div class="grid grid-cols-2 md:grid-cols-3 gap-2">
                <div v-for="s in wsState.stage_b?.steps" :key="s.id" class="studio-card p-2.5 rounded-lg flex items-center justify-between" :class="s.done ? 'border-l-4 border-emerald-500' : ''">
                  <span class="font-medium">{{ s.name }}</span>
                  <span>{{ s.done ? '✅' : (wsState.task?.stage === 'subtitle' && wsState.task?.step_index === s.id ? '⏳' : '⚪') }}</span>
                </div>
              </div>
            </div>

            <div class="studio-subtle p-3 rounded-xl flex items-center justify-between">
              <div>
                <span class="font-bold">📄 上传本地字幕文件跳过 WhisperX (.srt / .vtt)</span>
                <span class="text-slate-400 ml-2">直接复用已有时间轴字幕并自动执行去重叠清洗</span>
              </div>
              <label class="px-3 py-1.5 rounded-lg bg-blue-600/15 text-blue-600 dark:text-blue-400 font-semibold cursor-pointer hover:bg-blue-600 hover:text-white transition">
                <input type="file" class="hidden" accept=".srt,.vtt" @change="uploadSubtitleBypass" />
                📂 选择 .srt / .vtt 上传
              </label>
            </div>

            <div v-if="wsState.stage_b?.suspicious_terms?.length" class="studio-subtle p-3.5 rounded-xl flex flex-col gap-2">
              <div class="font-bold">🔍 识别到的可疑或高频专业词汇</div>
              <div class="overflow-x-auto max-h-48">
                <table class="w-full text-left border-collapse">
                  <thead>
                    <tr class="border-b border-slate-200 dark:border-slate-700 text-slate-400">
                      <th class="py-1.5 px-2">词汇</th>
                      <th class="py-1.5 px-2">分类</th>
                      <th class="py-1.5 px-2">频次</th>
                      <th class="py-1.5 px-2">上下文示例</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    <tr v-for="item in wsState.stage_b.suspicious_terms" :key="item.term">
                      <td class="py-1.5 px-2 font-mono font-semibold text-amber-500">{{ item.term }}</td>
                      <td class="py-1.5 px-2">{{ item.category }}</td>
                      <td class="py-1.5 px-2">{{ item.count }}</td>
                      <td class="py-1.5 px-2 text-slate-400 truncate max-w-md">{{ item.example }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-5 gap-2.5">
              <button @click="startSubtitlePipeline(false)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-blue-600 text-white font-bold hover:bg-blue-700 disabled:opacity-50 transition">
                ▶️ 开始处理字幕
              </button>
              <button v-if="wsState.stage_b?.src_srt_exists && !wsState.stage_b?.sub_video_exists" @click="startSubtitlePipeline(true)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-emerald-600 text-white font-semibold hover:bg-emerald-700 transition">
                🔥 烧录字幕到视频
              </button>
              <button @click="reapplyTerms" class="studio-subtle py-2.5 rounded-lg font-semibold hover:border-blue-500 transition">
                🔄 重新应用词表并清理
              </button>
              <a v-if="wsState.stage_b?.src_srt_exists" href="/api/subtitle/download-zip" class="studio-subtle py-2.5 rounded-lg font-semibold hover:border-blue-500 transition text-center">
                📥 下载所有 SRT 字幕
              </a>
              <button @click="resetSubtitles" class="studio-subtle py-2.5 rounded-lg font-semibold text-amber-600 hover:border-amber-500 transition">
                ♻️ 重新生成字幕
              </button>
              <button @click="archiveWorkspace" class="studio-subtle py-2.5 rounded-lg font-semibold text-emerald-600 hover:border-emerald-500 transition">
                🗂️ 归档到 History
              </button>
            </div>
          </div>

          <!-- STAGE C -->
          <div v-if="activeStage === 'c'" class="flex flex-col gap-3.5 text-xs">
            <div class="studio-subtle p-2.5 rounded-xl grid grid-cols-2 md:grid-cols-4 gap-2">
              <div v-for="s in wsState.stage_c?.steps" :key="s.id" class="studio-card p-2.5 rounded-lg flex items-center justify-between" :class="s.waiting ? 'border-l-4 border-amber-500' : (s.done ? 'border-l-4 border-emerald-500' : '')">
                <span :class="s.waiting ? 'font-bold text-amber-500' : 'font-medium'">{{ s.name }}</span>
                <span>{{ s.waiting ? '⏸️' : (s.done ? '✅' : '⏳') }}</span>
              </div>
            </div>

            <div v-if="speakerReview.available" class="studio-subtle p-3.5 rounded-xl flex flex-col gap-3.5 border border-blue-500/30">
              <div class="flex flex-wrap items-center justify-between gap-2">
                <div class="flex items-center gap-2.5">
                  <span class="text-sm font-bold">🎙️ 说话人识别结果与配音角色微调</span>
                  <span class="px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-semibold">
                    已启用跨角色边界隔离
                  </span>
                  <button @click="openSettings('dub')" class="text-blue-500 hover:underline font-medium">
                    ⚙️ 默认音色设置
                  </button>
                </div>

                <div class="studio-card p-0.5 rounded-lg flex items-center gap-1">
                  <button @click="speakerSubTab = 'cards'" :class="speakerSubTab === 'cards' ? 'bg-blue-600 text-white font-semibold' : 'text-slate-500'" class="px-3 py-1 rounded-md text-xs transition">
                    🎭 按说话人角色卡片 ({{ speakerReview.speakers.length }}人)
                  </button>
                  <button @click="speakerSubTab = 'table'" :class="speakerSubTab === 'table' ? 'bg-blue-600 text-white font-semibold' : 'text-slate-500'" class="px-3 py-1 rounded-md text-xs transition">
                    📝 逐句台词微调表 ({{ speakerReview.lines.length }}句)
                  </button>
                </div>
              </div>

              <div v-if="speakerSubTab === 'cards'" class="grid grid-cols-1 md:grid-cols-3 gap-3.5">
                <div v-for="spk in speakerReview.speakers" :key="spk.speaker_id" class="studio-card p-3.5 rounded-xl flex flex-col gap-2.5 border-t-4 shadow-sm" :class="spk.gender === 'male' ? 'border-blue-500' : 'border-pink-500'">
                  <div class="flex items-center justify-between">
                    <span class="font-bold text-sm flex items-center gap-1.5">
                      <span>{{ spk.gender === 'male' ? '👨' : '👩' }}</span>
                      <code class="font-mono">{{ spk.speaker_id }}</code>
                    </span>
                    <span class="px-2 py-0.5 rounded bg-blue-500/10 text-blue-500 font-bold">共 {{ spk.count }} 句</span>
                  </div>

                  <div @click="seekToTime(spk.sample_start, spk.speaker_id, spk.sample_text)" class="text-slate-400 italic truncate cursor-pointer hover:text-blue-500" :title="spk.sample_text">
                    💬 "{{ spk.sample_text }}" ({{ spk.sample_timecode }})
                  </div>

                  <div v-if="spk.clip_url" class="studio-subtle p-2 rounded-lg flex flex-col gap-1">
                    <span class="text-slate-400 text-xs">🎧 原片人声切片试听:</span>
                    <audio :src="spk.clip_url" controls class="w-full h-8"></audio>
                  </div>

                  <div class="grid grid-cols-2 gap-2 pt-1">
                    <div class="flex flex-col gap-1">
                      <span class="text-slate-400">角色性别</span>
                      <select v-model="spk.gender" class="studio-input px-2 py-1.5 rounded-md font-semibold">
                        <option value="male">👨 男声 (Male)</option>
                        <option value="female">👩 女声 (Female)</option>
                      </select>
                    </div>
                    <div class="flex flex-col gap-1">
                      <span class="text-slate-400">专属绑定音色</span>
                      <select v-model="spk.voice" class="studio-input px-2 py-1.5 rounded-md font-medium">
                        <option value="">(跟随默认男/女音色)</option>
                        <option v-for="(vid, vlabel) in speakerReview.voices" :key="vid" :value="vid">{{ vlabel }}</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              <div v-if="speakerSubTab === 'table'" class="overflow-x-auto max-h-96 studio-card rounded-lg">
                <table class="w-full text-left border-collapse">
                  <thead class="sticky top-0 studio-subtle">
                    <tr class="border-b border-slate-200 dark:border-slate-800 text-slate-400">
                      <th class="py-2 px-2.5">#</th>
                      <th class="py-2 px-2.5">时间轴 (点击跳转视频)</th>
                      <th class="py-2 px-2.5">说话人 ID</th>
                      <th class="py-2 px-2.5">性别</th>
                      <th class="py-2 px-2.5">专属音色</th>
                      <th class="py-2 px-2.5">配音台词</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                    <tr v-for="line in speakerReview.lines" :key="line.number" class="hover:bg-blue-500/5">
                      <td class="py-1.5 px-2.5 font-mono">{{ line.number }}</td>
                      <td @click="seekToTime(line.start_sec, line.speaker_id, line.text)" class="py-1.5 px-2.5 font-mono text-blue-500 font-medium cursor-pointer hover:underline whitespace-nowrap">
                        🎯 {{ line.time_range }}
                      </td>
                      <td class="py-1.5 px-2.5">
                        <input v-model="line.speaker_id" class="studio-input px-2 py-1 rounded font-mono w-28" />
                      </td>
                      <td class="py-1.5 px-2.5">
                        <select v-model="line.gender" class="studio-input px-2 py-1 rounded">
                          <option value="male">👨 male</option>
                          <option value="female">👩 female</option>
                        </select>
                      </td>
                      <td class="py-1.5 px-2.5">
                        <select v-model="line.voice" class="studio-input px-2 py-1 rounded w-44">
                          <option value="">(跟随默认)</option>
                          <option v-for="(vid, vlabel) in speakerReview.voices" :key="vid" :value="vid">{{ vlabel }}</option>
                        </select>
                      </td>
                      <td class="py-1.5 px-2.5">
                        <input v-model="line.text" class="studio-input w-full min-w-[260px] px-2 py-1 rounded" />
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-2.5 pt-1">
                <button @click="saveSpeakerOverrides(false)" class="py-2 rounded-lg studio-card font-semibold hover:border-blue-500 transition">
                  💾 仅保存说话人与音色修改 (清理变更句缓存)
                </button>
                <button @click="saveSpeakerOverrides(true)" :disabled="wsState.task?.running" class="py-2 rounded-lg bg-indigo-600 text-white font-semibold hover:bg-indigo-700 disabled:opacity-50 transition">
                  🔄 保存并增量重新合成配音
                </button>
              </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
              <button v-if="wsState.stage_c?.waiting_confirm" @click="startDubbing(true)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-blue-600 text-white font-bold shadow hover:bg-blue-700 transition">
                ▶️ 确认说话人并继续生成配音
              </button>
              <button v-else @click="startDubbing(false)" :disabled="wsState.task?.running" class="py-2.5 rounded-lg bg-blue-600 text-white font-bold shadow hover:bg-blue-700 disabled:opacity-50 transition">
                🎙️ 开始生成配音任务
              </button>
              <button @click="deleteDubbing" class="studio-subtle py-2.5 rounded-lg font-semibold text-rose-500 hover:border-rose-500 transition">
                🗑️ 删除配音文件重试
              </button>
              <button @click="archiveWorkspace" class="studio-subtle py-2.5 rounded-lg font-semibold text-emerald-600 hover:border-emerald-500 transition">
                🗂️ 归档到 History
              </button>
            </div>
          </div>
        </section>
      </main>

      <!-- ==================== 右侧信息栏 (3 Cols / 25% Width) ==================== -->
      <aside class="col-span-12 lg:col-span-3 flex flex-col gap-4">
        <div class="studio-card rounded-xl p-3.5 flex flex-col gap-2.5 shadow-sm">
          <div class="text-xs font-bold flex items-center justify-between">
            <span>🖼️ 视频封面与标题</span>
          </div>
          <div v-if="wsState.video?.cover_url" class="rounded-lg overflow-hidden border border-slate-200 dark:border-slate-800">
            <img :src="wsState.video.cover_url" class="w-full h-36 object-cover" />
          </div>
          <div v-else class="rounded-lg h-32 studio-subtle border-2 border-dashed flex flex-col items-center justify-center text-slate-400 text-xs">
            <span class="text-xl mb-1">🖼️</span>
            <span>暂未下载封面</span>
          </div>
          <div class="font-bold text-xs text-center pt-0.5 leading-snug">{{ wsState.video?.title || '暂未获取到视频标题' }}</div>
        </div>

        <div class="studio-card rounded-xl p-3.5 flex flex-col gap-2.5 shadow-sm flex-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold">📂 文件浏览 (output/)</span>
            <div class="flex gap-1">
              <button @click="fetchWorkspaceState(true)" class="studio-subtle px-2 py-0.5 rounded text-xs hover:border-blue-500">🔄 刷新</button>
              <button @click="expandedDirs = {}" class="studio-subtle px-2 py-0.5 rounded text-xs hover:border-blue-500">📁 全部收起</button>
              <button v-if="wsState.files?.length" @click="deleteVideo" class="studio-subtle px-2 py-0.5 rounded text-xs text-rose-500 hover:border-rose-500">🗑️ 清空</button>
            </div>
          </div>

          <div class="studio-subtle rounded-lg p-2.5 text-xs font-mono flex flex-col gap-1 max-h-80 overflow-y-auto">
            <div v-if="!wsState.files?.length" class="text-slate-400 text-center py-6 font-sans">暂无产出文件</div>
            <template v-for="node in wsState.files" :key="node.path">
              <div v-if="node.type === 'dir'">
                <button @click="toggleDir(node.path)" class="w-full text-left py-1 px-1.5 rounded hover:bg-blue-500/10 flex items-center justify-between font-semibold">
                  <span>{{ expandedDirs[node.path] ? '▼ 📂' : '▶ 📁' }} {{ node.name }}/</span>
                  <span class="text-slate-400">{{ node.count }}</span>
                </button>
                <div v-if="expandedDirs[node.path]" class="pl-4 flex flex-col gap-0.5 mt-0.5 border-l border-slate-200 dark:border-slate-700 ml-2">
                  <template v-for="sub in node.children" :key="sub.path">
                    <div v-if="sub.type === 'dir'">
                      <button @click="toggleDir(sub.path)" class="w-full text-left py-0.5 px-1 rounded hover:bg-blue-500/10 font-semibold">
                        {{ expandedDirs[sub.path] ? '▼ 📂' : '▶ 📁' }} {{ sub.name }}/ ({{ sub.count }})
                      </button>
                      <div v-if="expandedDirs[sub.path]" class="pl-4 flex flex-col gap-0.5">
                        <a v-for="f2 in sub.children" :key="f2.path" :href="f2.url" :download="f2.name" class="py-0.5 px-1 rounded hover:bg-blue-500/10 hover:text-blue-500 truncate block">
                          📄 {{ f2.name }} <span class="text-slate-400">({{ f2.size }})</span>
                        </a>
                      </div>
                    </div>
                    <a v-else :href="sub.url" :download="sub.name" class="py-0.5 px-1 rounded hover:bg-blue-500/10 hover:text-blue-500 truncate flex justify-between">
                      <span class="truncate">📄 {{ sub.name }}</span>
                      <span class="text-slate-400 ml-1 shrink-0">{{ sub.size }}</span>
                    </a>
                  </template>
                </div>
              </div>
              <a v-else :href="node.url" :download="node.name" class="py-1 px-1.5 rounded hover:bg-blue-500/10 hover:text-blue-500 flex items-center justify-between">
                <span class="truncate">📄 {{ node.name }}</span>
                <span class="text-slate-400 ml-1 shrink-0">{{ node.size }}</span>
              </a>
            </template>
          </div>
        </div>

        <div class="studio-card rounded-xl p-3.5 flex flex-col gap-2 shadow-sm">
          <div class="text-xs font-bold">🔍 视频编码对比 (FFprobe)</div>
          <div v-if="wsState.encoding?.length" class="overflow-x-auto">
            <table class="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr class="border-b border-slate-200 dark:border-slate-800 text-slate-400">
                  <th class="py-1 px-1.5">阶段</th>
                  <th class="py-1 px-1.5">编码</th>
                  <th class="py-1 px-1.5">分辨率</th>
                  <th class="py-1 px-1.5">体积</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
                <tr v-for="row in wsState.encoding" :key="row.Stage">
                  <td class="py-1.5 px-1.5 font-sans font-semibold">{{ row.Stage }}</td>
                  <td class="py-1.5 px-1.5">{{ row.VideoCodec }}/{{ row.AudioCodec }}</td>
                  <td class="py-1.5 px-1.5">{{ row.Resolution }}</td>
                  <td class="py-1.5 px-1.5">{{ row.Size }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="text-xs text-slate-400 py-2 text-center">暂无视频编码信息</div>
        </div>
      </aside>
    </div>

    <!-- ==================== 2-ALT. 批量队列监控视图 ==================== -->
    <div v-if="workspaceMode === 'batch'" class="flex-1 p-4 max-w-[1600px] w-full mx-auto flex flex-col gap-4">
      <div class="studio-card rounded-xl p-4 flex flex-col gap-3 shadow-sm">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-base font-bold">📦 批处理全局监控中心 (`batch/tasks_setting.xlsx`)</h2>
            <p class="text-xs text-slate-400">读取并监控批量任务表格状态</p>
          </div>
          <button @click="switchBatchMode" class="studio-subtle px-3 py-1.5 rounded-lg text-xs font-semibold hover:border-blue-500">🔄 刷新队列</button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs border-collapse">
            <thead>
              <tr class="border-b border-slate-200 dark:border-slate-800 text-slate-400">
                <th class="py-2 px-3">Video File / URL</th>
                <th class="py-2 px-3">Source Lang</th>
                <th class="py-2 px-3">Target Lang</th>
                <th class="py-2 px-3">Dubbing</th>
                <th class="py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-200 dark:divide-slate-800">
              <tr v-for="(t, i) in batchTasks" :key="i">
                <td class="py-2.5 px-3 font-mono">{{ t['Video File'] }}</td>
                <td class="py-2.5 px-3">{{ t['Source Language'] }}</td>
                <td class="py-2.5 px-3">{{ t['Target Language'] }}</td>
                <td class="py-2.5 px-3">{{ t['Dubbing'] }}</td>
                <td class="py-2.5 px-3 font-semibold">{{ t['Status'] || 'Pending' }}</td>
              </tr>
              <tr v-if="!batchTasks.length">
                <td colspan="5" class="py-8 text-center text-slate-400">暂无批量任务记录 (`batch/tasks_setting.xlsx`)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ==================== 3. 独立参数设置弹窗 ==================== -->
    <div v-if="showSettingsModal" class="fixed inset-0 z-50 bg-slate-950/60 backdrop-blur-xs flex items-center justify-center p-4" @click.self="showSettingsModal = false">
      <div class="studio-card rounded-2xl w-full max-w-4xl max-h-[88vh] flex flex-col shadow-2xl overflow-hidden border border-slate-200 dark:border-slate-700">
        <div class="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <span class="text-base">⚙️</span>
            <span class="font-bold text-sm">系统参数设置 (`config.yaml`)</span>
          </div>
          <div class="flex items-center gap-2">
            <button @click="saveAllConfig" class="px-4 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white hover:bg-blue-700 transition">
              💾 保存全部设置
            </button>
            <button @click="showSettingsModal = false" class="studio-subtle w-7 h-7 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-white">
              ✕
            </button>
          </div>
        </div>

        <div class="flex-1 grid grid-cols-12 overflow-hidden" v-if="cfg.api">
          <div class="col-span-3 border-r border-slate-200 dark:border-slate-800 p-3 flex flex-col gap-1.5 studio-subtle">
            <button @click="settingsTab = 'llm'" :class="settingsTab === 'llm' ? 'bg-blue-600 text-white shadow-xs' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-2.5 rounded-lg text-xs font-semibold text-left transition flex items-center gap-2">
              <span>🤖</span><span>LLM 与全局网络</span>
            </button>
            <button @click="settingsTab = 'sub'" :class="settingsTab === 'sub' ? 'bg-blue-600 text-white shadow-xs' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-2.5 rounded-lg text-xs font-semibold text-left transition flex items-center gap-2">
              <span>📝</span><span>字幕与转录设置</span>
            </button>
            <button @click="settingsTab = 'dub'" :class="settingsTab === 'dub' ? 'bg-blue-600 text-white shadow-xs' : 'text-slate-500 hover:bg-slate-500/10'" class="px-3 py-2.5 rounded-lg text-xs font-semibold text-left transition flex items-center gap-2">
              <span>🔊</span><span>配音与声纹设置</span>
            </button>
          </div>

          <div class="col-span-9 p-5 overflow-y-auto max-h-[74vh] text-xs">
            <!-- LLM -->
            <div v-if="settingsTab === 'llm'" class="flex flex-col gap-4">
              <label class="studio-subtle p-3 rounded-xl flex items-center justify-between cursor-pointer">
                <div>
                  <div class="font-bold text-sm">使用 Antigravity CLI (agy)</div>
                  <div class="text-slate-400 mt-0.5">启用本地命令行大模型直连通道</div>
                </div>
                <input type="checkbox" v-model="cfg.api.use_antigravity_cli" class="w-4 h-4 accent-blue-600" />
              </label>

              <div v-if="cfg.api.use_antigravity_cli" class="studio-subtle p-3 rounded-xl flex flex-col gap-1.5 border-l-4 border-amber-500">
                <label class="font-semibold">🔑 Antigravity 登录授权码 (Token Code)</label>
                <input type="text" v-model="cfg.api.antigravity_token_code" class="studio-input px-3 py-1.5 rounded-lg font-mono" />
              </div>

              <div v-else class="flex flex-col gap-3.5">
                <div class="studio-subtle p-3.5 rounded-xl flex flex-col gap-2.5">
                  <div class="font-bold">📂 LLM 配置预设 (Saved Profiles)</div>
                  <div class="flex gap-2">
                    <select v-model="selectedProfile" class="studio-input flex-1 px-2.5 py-1.5 rounded-lg">
                      <option value="">-- 选择已保存预设 --</option>
                      <option v-for="(_, name) in llmProfiles" :key="name" :value="name">{{ name }}</option>
                    </select>
                    <button @click="handleProfile('load')" class="studio-card px-3 py-1.5 rounded-lg font-medium hover:border-blue-500">📥 加载</button>
                    <button @click="handleProfile('delete')" class="studio-card px-3 py-1.5 rounded-lg font-medium text-rose-500 hover:border-rose-500">🗑️ 删除</button>
                  </div>
                  <div class="flex gap-2 pt-2 border-t border-slate-200 dark:border-slate-700">
                    <input v-model="newProfileName" type="text" placeholder="输入新预设名称..." class="studio-input flex-1 px-2.5 py-1.5 rounded-lg" />
                    <button @click="handleProfile('save')" class="px-3.5 py-1.5 rounded-lg bg-blue-600 text-white font-semibold">💾 保存新预设</button>
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-3">
                  <div class="flex flex-col gap-1">
                    <label class="font-medium text-slate-500">API_KEY</label>
                    <input type="password" v-model="cfg.api.key" class="studio-input px-3 py-1.5 rounded-lg font-mono" />
                  </div>
                  <div class="flex flex-col gap-1">
                    <label class="font-medium text-slate-500">BASE_URL</label>
                    <input type="text" v-model="cfg.api.base_url" class="studio-input px-3 py-1.5 rounded-lg font-mono" />
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-3 items-end">
                  <div class="flex flex-col gap-1">
                    <label class="font-medium text-slate-500">MODEL</label>
                    <div class="flex gap-1.5">
                      <input type="text" v-model="cfg.api.model" class="studio-input flex-1 px-3 py-1.5 rounded-lg font-mono" />
                      <button @click="testLlmApi" class="studio-subtle px-3 py-1.5 rounded-lg hover:border-blue-500 font-semibold">📡 测试</button>
                    </div>
                  </div>
                  <label class="studio-subtle px-3 py-2 rounded-lg flex items-center justify-between cursor-pointer">
                    <span>LLM JSON 格式强制输出支持</span>
                    <input type="checkbox" v-model="cfg.api.llm_support_json" class="w-4 h-4 accent-blue-600" />
                  </label>
                </div>
              </div>

              <div class="studio-subtle p-3.5 rounded-xl grid grid-cols-2 gap-3">
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">🌐 全局网络代理 (Proxy)</label>
                  <input type="text" v-model="cfg.proxy" placeholder="http://127.0.0.1:7897" class="studio-input px-3 py-1.5 rounded-lg font-mono" />
                </div>
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">🗣️ 界面语言</label>
                  <select v-model="cfg.display_language" class="studio-input px-3 py-1.5 rounded-lg">
                    <option value="zh-CN">简体中文</option>
                    <option value="en">English</option>
                    <option value="zh-HK">繁體中文</option>
                    <option value="ja">日本語</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- SUBTITLE -->
            <div v-if="settingsTab === 'sub'" class="flex flex-col gap-3.5">
              <div class="grid grid-cols-3 gap-3">
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">识别原语言</label>
                  <input type="text" v-model="cfg.whisper.language" class="studio-input px-2.5 py-1.5 rounded-lg" />
                </div>
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">翻译目标语言</label>
                  <input type="text" v-model="cfg.target_language" class="studio-input px-2.5 py-1.5 rounded-lg" />
                </div>
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">WhisperX 运行模式</label>
                  <select v-model="cfg.whisper.runtime" class="studio-input px-2.5 py-1.5 rounded-lg">
                    <option value="mlx">🍎 mlx (Apple Metal 加速)</option>
                    <option value="local">💻 local (本地 GPU/CPU)</option>
                    <option value="cloud">☁️ cloud (302.ai)</option>
                    <option value="elevenlabs">🎙️ elevenlabs</option>
                  </select>
                </div>
              </div>

              <div class="grid grid-cols-4 gap-2">
                <label class="studio-subtle p-2.5 rounded-lg flex items-center justify-between cursor-pointer">
                  <span>人声分离 (Demucs)</span>
                  <input type="checkbox" v-model="cfg.demucs" class="w-4 h-4 accent-blue-600" />
                </label>
                <label class="studio-subtle p-2.5 rounded-lg flex items-center justify-between cursor-pointer">
                  <span>FFmpeg GPU 加速</span>
                  <input type="checkbox" v-model="cfg.ffmpeg_gpu" class="w-4 h-4 accent-blue-600" />
                </label>
                <label class="studio-subtle p-2.5 rounded-lg flex items-center justify-between cursor-pointer">
                  <span>烧录字幕到视频</span>
                  <input type="checkbox" v-model="cfg.burn_subtitles" class="w-4 h-4 accent-blue-600" />
                </label>
                <label class="studio-subtle p-2.5 rounded-lg flex items-center justify-between cursor-pointer">
                  <span>完成后自动烧录</span>
                  <input type="checkbox" v-model="cfg.auto_burn_subtitles" class="w-4 h-4 accent-blue-600" />
                </label>
              </div>

              <div class="flex flex-col gap-1">
                <label class="font-semibold">📚 领域词汇表 (`domain_vocab` · 支持 `错误词 -> 正确词`)</label>
                <textarea v-model="customTermsText" rows="4" class="studio-input p-2.5 rounded-lg font-mono text-xs"></textarea>
              </div>

              <div class="grid grid-cols-2 gap-3">
                <div class="studio-subtle p-3 rounded-xl flex flex-col gap-2">
                  <label class="flex items-center justify-between cursor-pointer">
                    <span class="font-semibold">⚡ 效率模式 (大批量并发提交 LLM)</span>
                    <input type="checkbox" v-model="cfg.efficiency_mode" class="w-4 h-4 accent-blue-600" />
                  </label>
                  <div class="grid grid-cols-2 gap-2">
                    <div>
                      <span class="text-slate-400 block mb-0.5">批量分割句子数</span>
                      <input type="number" v-model.number="cfg.batch_split_size" class="studio-input w-full px-2 py-1 rounded" />
                    </div>
                    <div>
                      <span class="text-slate-400 block mb-0.5">批量翻译字幕数</span>
                      <input type="number" v-model.number="cfg.batch_translate_size" class="studio-input w-full px-2 py-1 rounded" />
                    </div>
                  </div>
                </div>

                <div class="studio-subtle p-3 rounded-xl flex flex-col gap-2" v-if="cfg.youtube">
                  <div class="font-semibold">🛡️ YouTube Cookies 管理</div>
                  <div class="flex items-center justify-between">
                    <span class="text-slate-500">从浏览器自动读取</span>
                    <select v-model="cfg.youtube.cookies_from_browser" class="studio-input px-2 py-1 rounded">
                      <option value="">不自动提取</option>
                      <option value="chrome">Chrome</option>
                      <option value="safari">Safari</option>
                      <option value="edge">Edge</option>
                      <option value="firefox">Firefox</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            <!-- DUBBING -->
            <div v-if="settingsTab === 'dub'" class="flex flex-col gap-3.5">
              <div class="grid grid-cols-2 gap-3">
                <div class="flex flex-col gap-1">
                  <label class="font-medium text-slate-500">TTS 语音合成引擎</label>
                  <select v-model="cfg.tts_method" class="studio-input px-3 py-1.5 rounded-lg font-semibold">
                    <option value="doubao_tts">🌋 火山引擎豆包 TTS 2.0</option>
                    <option value="edge_tts">🎙️ 微软 Edge TTS</option>
                    <option value="sf_cosyvoice2">🧬 SiliconFlow CosyVoice2</option>
                    <option value="gpt_sovits">🎭 GPT-SoVITS</option>
                  </select>
                </div>
                <div class="studio-subtle px-3 py-2 rounded-lg flex flex-col justify-center gap-1">
                  <div class="flex justify-between">
                    <span class="font-medium">TTS 批量并发线程数</span>
                    <span class="font-bold text-blue-600">{{ cfg.tts_max_workers }} 线程</span>
                  </div>
                  <input type="range" min="1" max="10" v-model.number="cfg.tts_max_workers" class="w-full accent-blue-600" />
                </div>
              </div>

              <div class="grid grid-cols-2 gap-3">
                <label class="studio-subtle p-2.5 rounded-xl flex items-center justify-between cursor-pointer">
                  <span>跳过提取原视频参考音频</span>
                  <input type="checkbox" v-model="cfg.skip_refer" class="w-4 h-4 accent-blue-600" />
                </label>
                <label class="p-2.5 rounded-xl flex items-center justify-between cursor-pointer bg-blue-500/10 border border-blue-500/30">
                  <div>
                    <span class="font-bold text-blue-600 dark:text-blue-400">⏸️ 配音前暂停确认说话人</span>
                    <span class="block text-slate-400 text-xs">第 1 步识别完自动暂停供试听与改音色</span>
                  </div>
                  <input type="checkbox" v-model="cfg.pause_before_dub" class="w-4 h-4 accent-blue-600" />
                </label>
              </div>

              <div class="grid grid-cols-2 gap-3">
                <div class="studio-subtle p-3 rounded-xl flex flex-col gap-2" v-if="cfg.pyannote">
                  <div class="flex items-center justify-between">
                    <span class="font-bold">🎙️ Pyannote 说话人声纹分离</span>
                    <span class="text-emerald-500 font-semibold">{{ cfg.pyannote.hf_token ? '✅ 已配置' : '未配置' }}</span>
                  </div>
                  <input type="password" v-model="cfg.pyannote.hf_token" placeholder="HuggingFace Token" class="studio-input px-2.5 py-1.5 rounded font-mono" />
                  <input type="text" v-model="cfg.pyannote.hf_endpoint" placeholder="HF 镜像站 (如 https://hf-mirror.com)" class="studio-input px-2.5 py-1.5 rounded font-mono" />
                  <div class="grid grid-cols-2 gap-2">
                    <div>
                      <span class="text-slate-400 block mb-0.5">最少说话人数</span>
                      <input type="number" v-model.number="cfg.pyannote.min_speakers" class="studio-input w-full px-2 py-1 rounded" />
                    </div>
                    <div>
                      <span class="text-slate-400 block mb-0.5">最多说话人数</span>
                      <input type="number" v-model.number="cfg.pyannote.max_speakers" class="studio-input w-full px-2 py-1 rounded" />
                    </div>
                  </div>
                </div>

                <div class="studio-subtle p-3 rounded-xl flex flex-col gap-2" v-if="cfg.doubao_tts">
                  <div class="font-bold">🎭 默认男女主音色配置</div>
                  <div class="grid grid-cols-2 gap-2">
                    <input type="text" v-model="cfg.doubao_tts.appid" placeholder="AppID" class="studio-input px-2.5 py-1 rounded font-mono" />
                    <input type="password" v-model="cfg.doubao_tts.access_token" placeholder="Access Token" class="studio-input px-2.5 py-1 rounded font-mono" />
                  </div>
                  <div class="flex flex-col gap-1">
                    <label class="text-slate-500 font-medium">👨 默认男声音色</label>
                    <select v-model="cfg.doubao_tts.male_voice" class="studio-input px-2 py-1.5 rounded-md">
                      <option v-for="(vid, vlabel) in voiceCatalogs.doubao_male" :key="vid" :value="vid">{{ vlabel }}</option>
                    </select>
                  </div>
                  <div class="flex flex-col gap-1">
                    <label class="text-slate-500 font-medium">👩 默认女声音色</label>
                    <select v-model="cfg.doubao_tts.female_voice" class="studio-input px-2 py-1.5 rounded-md">
                      <option v-for="(vid, vlabel) in voiceCatalogs.doubao_female" :key="vid" :value="vid">{{ vlabel }}</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ==================== 4. 底部可折叠实时终端日志抽屉 ==================== -->
    <footer class="sticky bottom-0 z-30 border-t border-slate-200 dark:border-slate-800" style="background-color: var(--console-bg); color: #e2e8f0;">
      <div class="px-4 py-1.5 flex items-center justify-between text-xs cursor-pointer" @click="showConsole = !showConsole">
        <div class="flex items-center gap-3 overflow-hidden">
          <span class="font-mono font-bold text-emerald-400 flex items-center gap-1.5 shrink-0">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            🖥️ 实时终端日志
          </span>
          <span class="text-slate-400 font-mono truncate">{{ wsState.logs?.length ? wsState.logs[wsState.logs.length - 1] : '等待任务输出...' }}</span>
        </div>
        <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-xs shrink-0">
          {{ showConsole ? '▼ 收起日志' : '▲ 展开日志' }}
        </span>
      </div>

      <div v-if="showConsole" class="px-4 py-3 border-t border-slate-800 font-mono text-xs max-h-48 overflow-y-auto flex flex-col gap-1 bg-slate-950">
        <div v-for="(line, idx) in wsState.logs" :key="idx" class="text-slate-300 leading-relaxed">{{ line }}</div>
      </div>
    </footer>

    <!-- Toast Notification -->
    <div v-if="toastMessage" class="fixed top-14 right-5 z-50 px-4 py-2.5 rounded-xl bg-slate-900/95 text-white text-xs font-semibold shadow-xl border border-white/10 backdrop-blur">
      {{ toastMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'

const isDark = ref(false)
const workspaceMode = ref('single')
const activeStage = ref('a')
const activeVideoSource = ref('orig')
const activeLineInfo = ref('')
const speakerSubTab = ref('cards')
const showSettingsModal = ref(false)
const settingsTab = ref('llm')
const showConsole = ref(false)
const toastMessage = ref('')
const videoPlayerRef = ref(null)

const ytbUrl = ref('')
const ytbRes = ref('1080')
const expandedDirs = reactive({ audio: true })

const cfg = reactive({})
const llmProfiles = ref({})
const selectedProfile = ref('')
const newProfileName = ref('')
const customTermsText = ref('')
const voiceCatalogs = reactive({ doubao_male: {}, doubao_female: {}, edge_male: {}, edge_female: {} })

const wsState = reactive({
  video: {},
  stage_b: { steps: [], suspicious_terms: [] },
  stage_c: { steps: [], speaker_review: { available: false, speakers: [], lines: [], voices: {} } },
  task: { running: false },
  files: [],
  encoding: [],
  logs: [],
})

const speakerReview = reactive({
  available: false,
  speakers: [],
  lines: [],
  voices: {},
})

const batchTasks = ref([])

function showToast(msg) {
  toastMessage.value = msg
  clearTimeout(window._tTimer)
  window._tTimer = setTimeout(() => { toastMessage.value = '' }, 3200)
}

function toggleTheme() {
  isDark.value = !isDark.value
  const html = document.documentElement
  html.classList.toggle('dark', isDark.value)
  html.classList.toggle('light', !isDark.value)
}

const currentVideoUrl = computed(() => {
  if (activeVideoSource.value === 'dub' && wsState.video?.dub_url) return wsState.video.dub_url
  if (activeVideoSource.value === 'sub' && wsState.video?.sub_url) return wsState.video.sub_url
  return wsState.video?.orig_url || null
})

function selectVideoSource(src) {
  activeVideoSource.value = src
}

function seekToTime(sec, speaker) {
  activeLineInfo.value = `${speaker} · ${sec}s`
  if (videoPlayerRef.value) {
    videoPlayerRef.value.currentTime = Number(sec) || 0
    videoPlayerRef.value.play().catch(() => {})
  }
  showToast(`🎯 已跳转至 ${sec}s · ${speaker}`)
}

function toggleDir(path) {
  expandedDirs[path] = !expandedDirs[path]
}

async function fetchConfig() {
  const res = await fetch('/api/config')
  const data = await res.json()
  Object.assign(cfg, data.config || {})
  llmProfiles.value = data.profiles || {}
  customTermsText.value = data.custom_terms || ''
  Object.assign(voiceCatalogs, data.voice_catalogs || {})
  if (cfg.ytb_resolution) ytbRes.value = String(cfg.ytb_resolution)
}

async function fetchWorkspaceState(syncSpeakerReview = false) {
  const res = await fetch('/api/workspace/state')
  const data = await res.json()
  wsState.video = data.video || {}
  wsState.stage_b = data.stage_b || {}
  wsState.stage_c = data.stage_c || {}
  wsState.task = data.task || {}
  wsState.files = data.files || []
  wsState.encoding = data.encoding || []
  if (data.logs?.length) wsState.logs = data.logs
  if (wsState.video.youtube_url && !ytbUrl.value) {
    ytbUrl.value = wsState.video.youtube_url
  }

  if (syncSpeakerReview || !speakerReview.available) {
    const sr = data.stage_c?.speaker_review
    if (sr && sr.available) {
      speakerReview.available = true
      speakerReview.speakers = sr.speakers
      speakerReview.lines = sr.lines
      speakerReview.voices = sr.voices
    } else {
      speakerReview.available = false
    }
  }
}

function openSettings(tab) {
  settingsTab.value = tab || 'llm'
  showSettingsModal.value = true
}

async function saveAllConfig() {
  const updates = {
    'api.use_antigravity_cli': Boolean(cfg.api?.use_antigravity_cli),
    'api.antigravity_token_code': cfg.api?.antigravity_token_code || '',
    'api.key': cfg.api?.key || '',
    'api.base_url': cfg.api?.base_url || '',
    'api.model': cfg.api?.model || '',
    'api.llm_support_json': Boolean(cfg.api?.llm_support_json),
    proxy: cfg.proxy || '',
    display_language: cfg.display_language || 'zh-CN',
    'whisper.language': cfg.whisper?.language || 'en',
    'whisper.runtime': cfg.whisper?.runtime || 'local',
    target_language: cfg.target_language || '简体中文',
    demucs: Boolean(cfg.demucs),
    ffmpeg_gpu: Boolean(cfg.ffmpeg_gpu),
    burn_subtitles: Boolean(cfg.burn_subtitles),
    auto_burn_subtitles: Boolean(cfg.auto_burn_subtitles),
    efficiency_mode: Boolean(cfg.efficiency_mode),
    batch_split_size: Number(cfg.batch_split_size ?? 100),
    batch_translate_size: Number(cfg.batch_translate_size ?? 15),
    'youtube.cookies_from_browser': cfg.youtube?.cookies_from_browser || '',
    tts_method: cfg.tts_method || 'doubao_tts',
    tts_max_workers: Number(cfg.tts_max_workers || 5),
    skip_refer: Boolean(cfg.skip_refer),
    pause_before_dub: Boolean(cfg.pause_before_dub),
    'pyannote.hf_token': cfg.pyannote?.hf_token || '',
    'pyannote.hf_endpoint': cfg.pyannote?.hf_endpoint || '',
    'pyannote.min_speakers': Number(cfg.pyannote?.min_speakers || 1),
    'pyannote.max_speakers': Number(cfg.pyannote?.max_speakers || 10),
    'doubao_tts.appid': cfg.doubao_tts?.appid || '',
    'doubao_tts.access_token': cfg.doubao_tts?.access_token || '',
    'doubao_tts.male_voice': cfg.doubao_tts?.male_voice || '',
    'doubao_tts.female_voice': cfg.doubao_tts?.female_voice || '',
  }
  await fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ updates, custom_terms: customTermsText.value }),
  })
  showSettingsModal.value = false
  showToast('💾 配置已保存并同步到 config.yaml')
  fetchWorkspaceState(true)
}

async function handleProfile(action) {
  const name = action === 'save' ? newProfileName.value.trim() : selectedProfile.value
  if (!name) return showToast('⚠️ 请先输入或选择预设名称')
  if (action === 'save') await saveAllConfig()
  const res = await fetch('/api/llm/profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, name }),
  })
  const data = await res.json()
  showToast(data.message || '操作成功')
  await fetchConfig()
}

async function testLlmApi() {
  await saveAllConfig()
  showToast('📡 正在检测 API 连通性...')
  const res = await fetch('/api/llm/check', { method: 'POST' })
  const data = await res.json()
  showToast(data.valid ? '✅ API 连接测试通过！' : '❌ API 连接失败，请检查密钥与地址')
}

function formatElapsed(sec) {
  const s = Math.max(0, Number(sec) || 0)
  const m = Math.floor(s / 60)
  const rem = s % 60
  return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`
}

async function downloadYoutube(subOnly) {
  if (!ytbUrl.value.trim()) return showToast('⚠️ 请输入 YouTube 视频链接')
  wsState.task = {
    running: true,
    stage: 'download',
    step_index: 1,
    total_steps: 4,
    step_name: subOnly ? '正在下载 YouTube 官方字幕...' : `正在下载 YouTube 视频 (${ytbRes.value}p)...`,
    sub_step: '正在启动 yt-dlp 解析链接与 Cookie 登录态...',
    percent: 3.0,
    speed: '',
    eta: '',
    total_size: '',
    elapsed_sec: 0,
  }
  await fetch('/api/video/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: ytbUrl.value, resolution: ytbRes.value, subtitle_only: subOnly }),
  })
  showToast(subOnly ? '⚡ 开始下载官方字幕...' : '📥 开始下载视频与字幕...')
  fetchWorkspaceState()
}

async function uploadVideoFile(e) {
  const file = e.target.files?.[0]
  if (!file) return
  const fd = new FormData()
  fd.append('file', file)
  showToast('📤 正在上传音视频文件...')
  await fetch('/api/video/upload', { method: 'POST', body: fd })
  showToast('✅ 本地文件上传完成！已自动切换至阶段 B')
  await fetchWorkspaceState(true)
  if (wsState.video?.exists) {
    activeStage.value = 'b'
  }
}

async function deleteVideo() {
  await fetch('/api/video', { method: 'DELETE' })
  ytbUrl.value = ''
  speakerReview.available = false
  speakerReview.speakers = []
  speakerReview.lines = []
  activeStage.value = 'a'
  showToast('🗑️ 已清空当前任务视频与 output/ 工作区文件')
  fetchWorkspaceState(true)
}

async function uploadSubtitleBypass(e) {
  const file = e.target.files?.[0]
  if (!file) return
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/subtitle/upload', { method: 'POST', body: fd })
  if (res.ok) {
    showToast('✅ 字幕解析成功！已跳过 WhisperX 步骤')
    fetchWorkspaceState()
  } else {
    showToast('❌ 字幕文件解析失败')
  }
}

async function startSubtitlePipeline(burnOnly) {
  wsState.task = {
    running: true,
    stage: 'subtitle',
    step_index: burnOnly ? 6 : 1,
    total_steps: 6,
    step_name: burnOnly ? '6. 正在烧录字幕到视频...' : '1. 正在准备字幕与三步翻译流水线...',
    sub_step: '正在初始化字幕流水线任务...',
    percent: burnOnly ? 85.0 : 5.0,
    speed: '',
    eta: '',
    total_size: '',
    elapsed_sec: 0,
  }
  await fetch(`/api/pipeline/subtitle/start?burn_only=${burnOnly}`, { method: 'POST' })
  showToast('🚀 字幕处理流水线已启动')
  fetchWorkspaceState()
}

async function reapplyTerms() {
  await fetch('/api/pipeline/subtitle/reapply-terms', { method: 'POST' })
  showToast('🔄 已重新应用领域词表并清空下游字幕缓存')
  fetchWorkspaceState()
}

async function resetSubtitles() {
  await fetch('/api/pipeline/subtitle/reset', { method: 'POST' })
  showToast('♻️ 已清空字幕翻译缓存')
  fetchWorkspaceState()
}

async function archiveWorkspace() {
  await fetch('/api/pipeline/archive', { method: 'POST' })
  showToast('🗂️ 已归档至 history/ 目录')
  fetchWorkspaceState(true)
}

async function startDubbing(forceContinue) {
  wsState.task = {
    running: true,
    stage: 'dubbing',
    step_index: forceContinue ? 3 : 1,
    total_steps: 4,
    step_name: forceContinue ? '3. 正在并发生成多角色配音并对齐时间轴...' : '1. 正在执行说话人性别识别与配音分块...',
    sub_step: '正在初始化多角色配音流水线...',
    percent: forceContinue ? 60.0 : 8.0,
    speed: '',
    eta: '',
    total_size: '',
    elapsed_sec: 0,
  }
  await fetch(`/api/pipeline/dubbing/start?force_continue=${forceContinue}`, { method: 'POST' })
  showToast('🎙️ 配音流水线已启动')
  fetchWorkspaceState(true)
}

async function saveSpeakerOverrides(rebuild) {
  const res = await fetch('/api/speakers/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      speakers: speakerReview.speakers,
      lines: speakerReview.lines,
      incremental_rebuild: rebuild,
    }),
  })
  const data = await res.json()
  showToast(`✅ 已保存角色修改（共清理 ${data.deleted_cache_files} 个受影响句子缓存）`)
  fetchWorkspaceState(true)
}

async function deleteDubbing() {
  await fetch('/api/pipeline/dubbing', { method: 'DELETE' })
  showToast('🗑️ 已清理配音缓存文件')
  fetchWorkspaceState(true)
}

async function stopTask() {
  await fetch('/api/task/stop', { method: 'POST' })
  showToast('⏹️ 已发送停止指令')
}

async function clearError() {
  await fetch('/api/task/clear-error', { method: 'POST' })
  fetchWorkspaceState()
}

async function switchBatchMode() {
  workspaceMode.value = 'batch'
  const res = await fetch('/api/batch/state')
  const data = await res.json()
  batchTasks.value = data.tasks || []
}

onMounted(async () => {
  await fetchConfig()
  await fetchWorkspaceState(true)
  if (wsState.stage_c?.waiting_confirm || speakerReview.available) {
    activeStage.value = 'c'
  } else if (wsState.video?.exists) {
    activeStage.value = 'b'
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${proto}//${location.host}/ws/logs`)
  let wasRunning = false
  let lastRunningStage = null
  let lastWorkspaceSyncTime = 0

  ws.onmessage = async (ev) => {
    try {
      const payload = JSON.parse(ev.data)
      if (payload.logs) wsState.logs = payload.logs
      if (payload.task) {
        if (payload.task.toast) showToast(payload.task.toast)

        if (payload.task.running) {
          lastRunningStage = payload.task.stage || lastRunningStage
          const now = Date.now()
          if (now - lastWorkspaceSyncTime > 2500) {
            lastWorkspaceSyncTime = now
            fetchWorkspaceState(false)
          }
        }

        if (wasRunning && !payload.task.running) {
          const finishedStage = lastRunningStage
          lastRunningStage = null
          await fetchWorkspaceState(true)
          if (finishedStage === 'download' && !payload.task.error && wsState.video?.exists) {
            activeStage.value = 'b'
            showToast('🎬 视频与字幕准备就绪，已为您自动切换至「阶段 B · 字幕与翻译」')
          } else if (finishedStage === 'subtitle' && !payload.task.error && wsState.stage_b?.src_srt_exists) {
            activeStage.value = 'c'
          }
        }

        wasRunning = payload.task.running
        wsState.task = payload.task
      }
    } catch (_) {}
  }
})
</script>

<style>
:root, html.light {
  --bg-app: #f4f6fb;
  --bg-panel: #ffffff;
  --bg-subtle: #f8fafc;
  --bg-elevated: #f1f5f9;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --border-color: #e2e8f0;
  --accent: #2563eb;
  --accent-soft: rgba(37, 99, 235, 0.1);
  --console-bg: #0f172a;
}
html.dark {
  --bg-app: #090d16;
  --bg-panel: #111827;
  --bg-subtle: #1e293b;
  --bg-elevated: #1f2937;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --border-color: #1e293b;
  --accent: #3b82f6;
  --accent-soft: rgba(59, 130, 246, 0.15);
  --console-bg: #030712;
}
body {
  background-color: var(--bg-app);
  color: var(--text-main);
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", "Microsoft YaHei", sans-serif;
  transition: background-color 0.2s ease, color 0.2s ease;
}
.studio-card {
  background-color: var(--bg-panel);
  border: 1px solid var(--border-color);
}
.studio-subtle {
  background-color: var(--bg-subtle);
  border: 1px solid var(--border-color);
}
.studio-input {
  background-color: var(--bg-subtle);
  border: 1px solid var(--border-color);
  color: var(--text-main);
}
.studio-input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--accent-soft);
}
</style>
