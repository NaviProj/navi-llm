//! LLM 配置模块

use std::num::NonZeroU32;
use std::path::PathBuf;

/// 采样模式
#[derive(Debug, Clone)]
pub enum SamplingMode {
    /// 一般任务：min_p → temp → dist，有随机性，适合聊天、写作
    Creative { temperature: f32, min_p: f32 },
    /// Agent 任务：极低温度 + min_p，平衡确定性和避免退化循环
    /// 适合 tool calling、结构化输出中仍需少量自然语言推理的场景
    Agent { temperature: f32, min_p: f32 },
    /// 纯贪心解码，完全确定性输出
    Deterministic,
    /// 自定义采样参数
    Custom {
        temperature: f32,
        min_p: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<i32>,
        seed: u32,
    },
}

impl Default for SamplingMode {
    fn default() -> Self {
        SamplingMode::Creative {
            temperature: 0.7,
            min_p: 0.05,
        }
    }
}

/// LLM 模型配置
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// GGUF 模型文件路径
    pub model_path: PathBuf,
    /// 视觉投影模型 (mmproj) 文件路径，启用视觉能力
    pub mmproj_path: Option<PathBuf>,
    /// 上下文大小 (默认 2048)
    pub ctx_size: NonZeroU32,
    /// 最大生成 token 数 (默认 1024)
    pub max_tokens: u32,
    /// 推理线程数 (None 表示使用所有可用线程)
    pub n_threads: Option<i32>,
    /// 批处理线程数 (None 表示使用所有可用线程)
    pub n_threads_batch: Option<i32>,
    /// 随机种子
    pub seed: u32,
    /// 是否启用详细日志
    pub verbose: bool,
    /// 系统提示词
    pub system_prompt: Option<String>,
    /// 是否启用思考模式 (Qwen3.5 等模型的 <think> 标签)
    pub enable_thinking: bool,
    /// 采样模式
    pub sampling_mode: SamplingMode,
    /// 是否启用 8-bit 量化 KV Cache (Q8_0)，可显著减少内存占用
    pub kv_cache_q8: bool,
    /// GPU offload 层数，None 表示使用默认值（不 offload）
    /// 设置为 u32::MAX 表示全部 offload 到 GPU
    pub n_gpu_layers: Option<u32>,
    /// 最大序列数，None = 默认 1。设置 >= 2 时启用 KV cache checkpoint 功能，
    /// 允许在 seq 1 中保存 system prompt 的 KV cache 快照，
    /// 使 hybrid 架构模型（如 Qwen3.5）也能复用 system prompt 缓存。
    pub n_seq_max: Option<u32>,
}

impl LlmConfig {
    /// 创建新配置
    pub fn new<P: Into<PathBuf>>(model_path: P) -> Self {
        Self {
            model_path: model_path.into(),
            mmproj_path: None,
            ctx_size: NonZeroU32::new(2048).unwrap(),
            max_tokens: 1024,
            n_threads: None,
            n_threads_batch: None,
            seed: 1234,
            verbose: false,
            system_prompt: None,
            enable_thinking: true,
            sampling_mode: SamplingMode::default(),
            kv_cache_q8: false,
            n_gpu_layers: None,
            n_seq_max: None,
        }
    }

    /// 设置上下文大小
    pub fn with_ctx_size(mut self, size: u32) -> Self {
        if let Some(s) = NonZeroU32::new(size) {
            self.ctx_size = s;
        } else {
            tracing::warn!(
                "with_ctx_size(0) 被忽略：上下文大小必须大于 0，保持当前值 {}",
                self.ctx_size
            );
        }
        self
    }

    /// 设置最大生成 token 数
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    /// 设置线程数
    pub fn with_threads(mut self, n: i32) -> Self {
        self.n_threads = Some(n);
        self.n_threads_batch = Some(n);
        self
    }

    /// 设置随机种子
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// 启用详细日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 设置系统提示词
    pub fn with_system_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// 设置是否启用思考模式
    ///
    /// 当设置为 false 时，Qwen3.5 等支持思考模式的模型将跳过 <think> 推理过程，
    /// 直接生成最终回答，减少延迟和 token 消耗。
    pub fn with_enable_thinking(mut self, enable: bool) -> Self {
        self.enable_thinking = enable;
        self
    }

    /// 设置采样模式
    pub fn with_sampling_mode(mut self, mode: SamplingMode) -> Self {
        self.sampling_mode = mode;
        self
    }

    /// 设置为 Creative 采样模式
    pub fn with_sampling_creative(mut self, temperature: f32, min_p: f32) -> Self {
        self.sampling_mode = SamplingMode::Creative { temperature, min_p };
        self
    }

    /// 设置为 Deterministic 采样模式 (greedy)
    pub fn with_sampling_deterministic(mut self) -> Self {
        self.sampling_mode = SamplingMode::Deterministic;
        self
    }

    /// 设置为 Agent 采样模式 (低温 + min_p)
    ///
    /// 默认 temperature=0.15, min_p=0.05。
    /// 适合 tool calling、结构化输出等需要高确定性但仍有少量推理的场景。
    pub fn with_sampling_agent(mut self) -> Self {
        self.sampling_mode = SamplingMode::Agent {
            temperature: 0.15,
            min_p: 0.05,
        };
        self
    }

    /// 设置为 Agent 采样模式（自定义参数）
    pub fn with_sampling_agent_params(mut self, temperature: f32, min_p: f32) -> Self {
        self.sampling_mode = SamplingMode::Agent { temperature, min_p };
        self
    }

    /// 设置视觉投影模型路径
    pub fn with_mmproj<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.mmproj_path = Some(path.into());
        self
    }

    /// 启用 8-bit 量化 KV Cache (Q8_0)
    ///
    /// 开启后 K Cache 和 V Cache 均使用 Q8_0 量化，
    /// 可大幅降低 KV Cache 内存占用（约为 F16 的一半），对生成质量影响极小。
    pub fn with_kv_cache_q8(mut self, enable: bool) -> Self {
        self.kv_cache_q8 = enable;
        self
    }

    /// 设置 GPU offload 层数
    ///
    /// - `None`: 不 offload（纯 CPU）
    /// - `Some(n)`: offload n 层到 GPU
    /// - `Some(u32::MAX)`: 全部 offload 到 GPU
    ///
    /// 需要编译时开启 `cuda`（NVIDIA）或 `metal`（Apple）feature。
    pub fn with_n_gpu_layers(mut self, n: u32) -> Self {
        self.n_gpu_layers = Some(n);
        self
    }

    /// 设置最大序列数（>= 2 启用 KV cache checkpoint）
    pub fn with_n_seq_max(mut self, n: u32) -> Self {
        self.n_seq_max = Some(n);
        self
    }
}
