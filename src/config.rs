//! LLM 配置模块

use std::num::NonZeroU32;
use std::path::PathBuf;

/// LLM 模型配置
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// GGUF 模型文件路径
    pub model_path: PathBuf,
    /// 视觉投影模型 (mmproj) 文件路径，启用视觉能力
    pub mmproj_path: Option<PathBuf>,
    /// 上下文大小 (默认 2048)
    pub ctx_size: NonZeroU32,
    /// 最大生成 token 数 (默认 128)
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
    /// GBNF 语法字符串，用于约束输出格式 (例如强制 JSON 输出)
    pub grammar: Option<String>,
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
            grammar: None,
        }
    }

    /// 设置上下文大小
    pub fn with_ctx_size(mut self, size: u32) -> Self {
        if let Some(s) = NonZeroU32::new(size) {
            self.ctx_size = s;
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

    /// 设置 GBNF 语法
    pub fn with_grammar<S: Into<String>>(mut self, grammar: S) -> Self {
        self.grammar = Some(grammar.into());
        self
    }

    /// 设置视觉投影模型路径
    pub fn with_mmproj<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.mmproj_path = Some(path.into());
        self
    }
}
