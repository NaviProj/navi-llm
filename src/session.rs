//! LLM 会话模块
//!
//! `LlmSessionFactory` 加载模型，`ManagedSession` 管理单次对话上下文。
//! 使用增量编码方案，复用 KV Cache，避免重复 prefilling。

use anyhow::{Context, Result};
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{ChatTemplateResult, GrammarTriggerType, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::openai::{ChatParseStateOaicompat, OpenAIChatTemplateParams};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::config::LlmConfig;

/// 对话消息角色
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// 对话消息
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Option<serde_json::Value>,
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: serde_json::Value) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// LLM 会话统计信息
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// 当前上下文中已使用的 token 数
    pub tokens_used: i32,
    /// 上下文总容量
    pub ctx_size: u32,
    /// 对话轮次数
    pub turn_count: usize,
    /// 总生成 token 数
    pub total_generated: usize,
    /// 总 prefill token 数（衡量复用效率）
    pub total_prefilled: usize,
    /// 缓存命中的 token 数
    pub cache_hits: usize,
}

/// 会话工厂 - 用于创建和管理会话
///
/// 持有模型和后端，可以创建多个独立的会话。
pub struct LlmSessionFactory {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    config: LlmConfig,
    chat_template: LlamaChatTemplate,
    /// 可选的多模态上下文 (视觉投影)
    mtmd_ctx: Option<MtmdContext>,
}

impl LlmSessionFactory {
    /// 创建会话工厂
    pub fn new(config: LlmConfig) -> Result<Self> {
        let backend = Arc::new(LlamaBackend::init().context("无法初始化 llama 后端")?);
        Self::new_with_backend(backend, config)
    }

    /// 使用已有的 LlamaBackend 创建会话工厂（避免重复初始化后端）
    pub fn new_with_backend(backend: Arc<LlamaBackend>, config: LlmConfig) -> Result<Self> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(config.verbose));

        tracing::info!("正在加载本地 LLM 模型 (Factory): {:?}", config.model_path);
        let start = std::time::Instant::now();

        // 如果配置了 mmproj，使用 GPU 加速加载模型（视觉推理需要）
        let mut model_params = LlamaModelParams::default();
        if config.mmproj_path.is_some() {
            model_params = model_params.with_n_gpu_layers(1_000_000);
        }
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .with_context(|| format!("无法加载模型: {:?}", config.model_path))?;

        let duration = start.elapsed();
        tracing::info!("本地 LLM 模型加载成功，耗时: {:?}", duration);

        // 如果配置了 mmproj，加载多模态视觉投影上下文
        let mtmd_ctx = if let Some(ref mmproj_path) = config.mmproj_path {
            tracing::info!("正在加载视觉投影模型: {:?}", mmproj_path);
            let mtmd_start = std::time::Instant::now();
            let mtmd_params = MtmdContextParams {
                use_gpu: true,
                print_timings: false,
                n_threads: config.n_threads.unwrap_or(4),
                media_marker: CString::new(
                    llama_cpp_2::mtmd::mtmd_default_marker().to_string(),
                )?,
            };
            let ctx = MtmdContext::init_from_file(
                &mmproj_path.to_string_lossy(),
                &model,
                &mtmd_params,
            )
            .with_context(|| format!("无法加载视觉投影模型: {:?}", mmproj_path))?;
            tracing::info!(
                "视觉投影模型加载成功，耗时: {:?}",
                mtmd_start.elapsed()
            );
            Some(ctx)
        } else {
            None
        };

        let chat_template = model.chat_template(None).unwrap_or_else(|_| {
            tracing::info!("模型未提供聊天模板，使用默认的 chatml 模板");
            LlamaChatTemplate::new("chatml").expect("valid default template")
        });

        Ok(Self {
            backend,
            model,
            config,
            chat_template,
            mtmd_ctx,
        })
    }

    /// 创建新的独立会话
    pub fn create_session(&self) -> Result<ManagedSession<'_>> {
        ManagedSession::new(
            &self.backend,
            &self.model,
            self.config.clone(),
            self.chat_template.clone(),
        )
    }

    /// 创建带自定义选项的独立会话
    ///
    /// 允许覆盖 ctx_size 和 max_tokens
    pub fn create_session_with_options(
        &self,
        ctx_size: Option<u32>,
        max_tokens: Option<u32>,
    ) -> Result<ManagedSession<'_>> {
        let mut config = self.config.clone();
        if let Some(s) = ctx_size {
            if let Some(nz) = std::num::NonZeroU32::new(s) {
                config.ctx_size = nz;
            }
        }
        if let Some(t) = max_tokens {
            config.max_tokens = t;
        }

        ManagedSession::new(
            &self.backend,
            &self.model,
            config,
            self.chat_template.clone(),
        )
    }

    /// 获取模型信息
    pub fn model_info(&self) -> String {
        format!(
            "模型: {:?}, ctx_size: {}",
            self.config.model_path.file_name().unwrap_or_default(),
            self.config.ctx_size
        )
    }

    /// 获取配置
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// 获取共享的后端引用
    pub fn backend(&self) -> &Arc<LlamaBackend> {
        &self.backend
    }

    /// 是否具备视觉能力
    pub fn has_vision(&self) -> bool {
        self.mtmd_ctx.is_some()
    }

    /// 使用视觉能力进行图像+文本推理（流式）
    ///
    /// 仅当 factory 加载了 mmproj 时可用。
    pub fn complete_with_image_bytes_streaming<F>(
        &self,
        prompt: &str,
        image_bytes: &[u8],
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let mtmd_ctx = self
            .mtmd_ctx
            .as_ref()
            .context("视觉能力未启用: 未配置 mmproj_path")?;

        let bitmap = MtmdBitmap::from_buffer(mtmd_ctx, image_bytes)
            .map_err(|e| anyhow::anyhow!("无法解码图像: {:?}", e))?;

        let n_threads = self.config.n_threads.unwrap_or(4);
        let mut ctx_params = LlamaContextParams::default()
            .with_n_threads(n_threads)
            .with_n_ctx(Some(self.config.ctx_size));

        if self.config.kv_cache_q8 {
            ctx_params = ctx_params
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);
        }

        let mut context = self.model.new_context(&self.backend, ctx_params)?;

        // 如果 prompt 中没有图像标记，追加默认标记
        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let mut full_prompt = prompt.to_string();
        if !full_prompt.contains(&default_marker) {
            full_prompt.push_str(&default_marker);
        }

        let bitmap_refs = vec![&bitmap];

        let messages_json = serde_json::json!([{"role": "user", "content": full_prompt}]);
        let messages_json_str = serde_json::to_string(&messages_json)?;

        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json_str,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: self.config.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };

        let result = self
            .model
            .apply_chat_template_oaicompat(&self.chat_template, &params)?;
        let formatted_prompt = result.prompt;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        let n_past = chunks.eval_chunks(mtmd_ctx, &mut context, 0, 0, 1, true)?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        let mut batch = LlamaBatch::new(self.config.ctx_size.get() as usize, 1);
        let mut current_past = n_past;

        for _ in 0..self.config.max_tokens {
            let token = sampler.sample(&context, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece_bytes = self.model.token_to_piece_bytes(token, 512, true, None)?;
            let mut token_str = String::with_capacity(32);
            let _ = decoder.decode_to_string(&piece_bytes, &mut token_str, false);

            callback(&token_str);
            output.push_str(&token_str);

            batch.clear();
            batch.add(token, current_past, &[0], true)?;
            current_past += 1;

            context.decode(&mut batch)?;
        }

        Ok(output)
    }
}

/// 托管会话 - 管理单次对话的完整上下文
///
/// 使用增量编码方案：
/// - 首轮对话：编码完整 prompt，缓存到 KV Cache
/// - 后续轮次：只编码新增的 tokens，复用已有的 KV Cache
/// - 生成的 tokens 也会被缓存，供下一轮使用
pub struct ManagedSession<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    config: LlmConfig,
    chat_template: LlamaChatTemplate,
    /// 对话历史（用于构建完整 prompt）
    messages: Vec<ChatMessage>,
    /// 当前 KV Cache 中已编码的 token 数（即 n_past）
    n_past: i32,
    /// 已编码的 tokens 列表（用于增量对比）
    encoded_tokens: Vec<LlamaToken>,
    /// 采样器
    sampler: LlamaSampler,
    /// 统计信息
    stats: SessionStats,
    tools_json: Option<String>,
    /// External cancellation flag — checked during token generation
    cancel: Option<Arc<AtomicBool>>,
}

fn regex_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '.' | '^' | '$' | '|' | '(' | ')' | '*' | '+' | '?' | '[' | ']' | '{' | '}' | '\\' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn anchor_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_string();
    }
    let mut anchored = String::new();
    if !pattern.starts_with('^') {
        anchored.push('^');
    }
    anchored.push_str(pattern);
    if !pattern.ends_with('$') {
        anchored.push('$');
    }
    anchored
}

impl<'a> ManagedSession<'a> {
    fn new(
        backend: &LlamaBackend,
        model: &'a LlamaModel,
        config: LlmConfig,
        chat_template: LlamaChatTemplate,
    ) -> Result<Self> {
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(config.ctx_size));

        if config.kv_cache_q8 {
            ctx_params = ctx_params
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);
        }

        if let Some(threads) = config.n_threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = config.n_threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        tracing::debug!("正在创建新的 LLM 上下文 (ctx_size={})", config.ctx_size);
        let ctx = model
            .new_context(backend, ctx_params)
            .context("无法创建 llama 上下文")?;
        tracing::debug!("LLM 上下文创建成功");

        let ctx_size = ctx.n_ctx();

        // Sampler: 使用 dist 采样器
        // 注意: Grammar 在当前 llama-cpp-2 版本中有兼容性问题，暂不使用
        let _ = config.grammar.as_ref(); // 避免 unused 警告
        let sampler = LlamaSampler::dist(config.seed);

        let mut messages = Vec::new();
        if let Some(ref system_prompt) = config.system_prompt {
            messages.push(ChatMessage::system(system_prompt.clone()));
        }

        Ok(Self {
            ctx,
            model,
            config,
            chat_template,
            messages,
            n_past: 0,
            encoded_tokens: Vec::new(),
            sampler,
            stats: SessionStats {
                tokens_used: 0,
                ctx_size,
                turn_count: 0,
                total_generated: 0,
                total_prefilled: 0,
                cache_hits: 0,
            },
            tools_json: None,
            cancel: None,
        })
    }

    /// 发送用户消息并获取回复（非流式）
    pub fn chat(&mut self, query: &str) -> Result<String> {
        self.chat_impl(Some(query), None::<fn(&str)>)
    }

    /// 发送用户消息并获取回复（流式）
    pub fn chat_streaming<F>(&mut self, query: &str, callback: F) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.chat_impl(Some(query), Some(callback))
    }

    /// 基于当前历史生成回复（流式）
    pub fn complete_chat_streaming<F>(&mut self, callback: F) -> Result<String>
    where
        F: FnMut(&str),
    {
        self.chat_impl(None, Some(callback))
    }

    /// 内部实现：处理对话（增量编码）
    fn chat_impl<F>(&mut self, query: Option<&str>, mut callback: Option<F>) -> Result<String>
    where
        F: FnMut(&str),
    {
        let start_time = std::time::Instant::now();
        // 1. 添加用户消息到历史
        if let Some(q) = query {
            tracing::info!("收到本地 LLM 查询: {}", q);
            self.messages.push(ChatMessage::user(q));
        } else {
            tracing::info!("启动本地 LLM 补全 (基于现有历史)");
        }

        // 2. 构建完整 prompt 并分词
        let result = self.build_prompt()?;
        let prompt = result.prompt.clone();
        let new_tokens = self
            .model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Never)
            .context("分词失败")?;

        // 3. 计算增量：找出需要新编码的 tokens
        // 比较新 tokens 和已编码的 tokens，找出公共前缀长度
        let common_prefix_len = self.find_common_prefix(&new_tokens);

        tracing::debug!(
            "增量编码: new_tokens={}, encoded_tokens={}, common_prefix={}",
            new_tokens.len(),
            self.encoded_tokens.len(),
            common_prefix_len
        );

        // 如果公共前缀小于已编码的长度，说明有冲突
        // 尝试部分清除 KV Cache 保留公共前缀，失败则全量清除
        if common_prefix_len < self.encoded_tokens.len() {
            let partial_ok = self
                .ctx
                .clear_kv_cache_seq(
                    Some(0),
                    Some(common_prefix_len as u32),
                    Some(self.encoded_tokens.len() as u32),
                )
                .unwrap_or(false);

            if partial_ok {
                tracing::info!(
                    "部分清除 KV Cache [{}, {}), 保留公共前缀 {} tokens",
                    common_prefix_len,
                    self.encoded_tokens.len(),
                    common_prefix_len
                );
                self.n_past = common_prefix_len as i32;
                self.encoded_tokens.truncate(common_prefix_len);
            } else {
                tracing::warn!(
                    "部分清除 KV Cache 失败, 全量清除并重新编码 (公共前缀: {}, 已编码: {})",
                    common_prefix_len,
                    self.encoded_tokens.len()
                );
                self.ctx.clear_kv_cache();
                self.n_past = 0;
                self.encoded_tokens.clear();
            }
        }

        // 需要编码的新 tokens
        let tokens_to_encode: Vec<LlamaToken> = if self.encoded_tokens.len() < new_tokens.len() {
            new_tokens[self.encoded_tokens.len()..].to_vec()
        } else {
            // 如果已编码的 tokens 反而更多或相等，这是异常情况
            // 强制编码最后一个新 token 以获取 logits
            tracing::warn!(
                "异常状态: encoded_tokens({}) >= new_tokens({}), 强制编码",
                self.encoded_tokens.len(),
                new_tokens.len()
            );
            if let Some(&last) = new_tokens.last() {
                vec![last]
            } else {
                anyhow::bail!("无法生成：prompt 为空");
            }
        };

        // 4. 检查上下文容量
        let required_ctx = new_tokens.len() as i32 + self.config.max_tokens as i32;
        let n_ctx = self.ctx.n_ctx() as i32;
        if required_ctx > n_ctx {
            anyhow::bail!(
                "上下文容量不足: 需要 {} tokens，但只有 {} tokens 可用。考虑调用 clear() 清理历史。",
                required_ctx,
                n_ctx
            );
        }

        // 5. 编码新增的 tokens（增量 prefill） - 分批处理
        let mut batch = LlamaBatch::new(512, 1);
        if !tokens_to_encode.is_empty() {
            let batch_size = 512;
            for chunk in tokens_to_encode.chunks(batch_size) {
                batch.clear();
                let chunk_len = chunk.len();

                for (i, token) in chunk.iter().enumerate() {
                    let pos = self.n_past + i as i32;
                    let current_global_pos = self.n_past + i as i32;
                    let target_last_pos =
                        (self.encoded_tokens.len() + tokens_to_encode.len()) as i32 - 1;
                    let is_last_in_sequence = current_global_pos == target_last_pos;

                    batch.add(*token, pos, &[0], is_last_in_sequence)?;
                }

                self.ctx.decode(&mut batch).context("增量 prefill 失败")?;
                self.n_past += chunk_len as i32;
            }

            self.encoded_tokens.extend(tokens_to_encode.iter().cloned());
            self.stats.total_prefilled += tokens_to_encode.len();
        } else {
            // 如果没有新增 tokens，需要重新编码最后一个 token 来获取 logits
            // 这种情况理论上不应该发生（每轮至少会有新的 user 消息）
            // 但为安全起见，重新 decode 最后一个 token
            if let Some(&last_token) = self.encoded_tokens.last() {
                let last_pos = self.n_past - 1;
                batch.add(last_token, last_pos, &[0], true)?;
                self.ctx
                    .decode(&mut batch)
                    .context("重新编码最后 token 失败")?;
            } else {
                anyhow::bail!("会话状态异常：没有已编码的 tokens");
            }
        }

        // 记录缓存命中
        self.stats.cache_hits += common_prefix_len;

        // 6. 准备生成
        let mut oai_parser: Option<ChatParseStateOaicompat> =
            if result.parse_tool_calls {
                match result.streaming_state_oaicompat() {
                    Ok(state) => Some(state),
                    Err(e) => {
                        tracing::warn!("无法初始化 OAI 解析器: {:?}", e);
                        None
                    }
                }
            } else {
                None
            };

        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let max_tokens = self.config.max_tokens as i32;
        let n_len = self.n_past + max_tokens;

        // 采样索引：decode 后 logits 在 batch 的最后一个 token 位置
        // batch.n_tokens() - 1 就是正确的索引（此时 batch 一定非空）
        let mut logit_idx = batch.n_tokens() - 1;

        // 构建采样链：如果模板提供了 grammar，则使用 grammar 采样器
        let mut grammar_sampler = if let Some(grammar) = result.grammar.as_deref() {
            tracing::info!("使用 Grammar 采样器进行生成");
            let gs = if result.grammar_lazy {
                let mut preserved = std::collections::HashSet::new();
                for token_str in &result.preserved_tokens {
                    let tokens = self.model.str_to_token(token_str, llama_cpp_2::model::AddBos::Never).unwrap_or_default();
                    if tokens.len() == 1 {
                        preserved.insert(tokens[0]);
                    }
                }

                let mut trigger_tokens = Vec::new();
                let mut trigger_patterns = Vec::new();
                for trigger in &result.grammar_triggers {
                    match trigger.trigger_type {
                        GrammarTriggerType::Token => {
                            if let Some(token) = trigger.token {
                                trigger_tokens.push(token);
                            }
                        }
                        GrammarTriggerType::Word => {
                            let tokens = self.model.str_to_token(&trigger.value, llama_cpp_2::model::AddBos::Never).unwrap_or_default();
                            if tokens.len() == 1 && preserved.contains(&tokens[0]) {
                                trigger_tokens.push(tokens[0]);
                            } else {
                                trigger_patterns.push(regex_escape(&trigger.value));
                            }
                        }
                        GrammarTriggerType::Pattern => {
                            trigger_patterns.push(trigger.value.clone());
                        }
                        GrammarTriggerType::PatternFull => {
                            trigger_patterns.push(anchor_pattern(&trigger.value));
                        }
                    }
                }
                LlamaSampler::grammar_lazy_patterns(
                    &self.model,
                    grammar,
                    "root",
                    &trigger_patterns,
                    &trigger_tokens,
                )?
            } else {
                LlamaSampler::grammar(&self.model, grammar, "root")?
            };
            // 将 grammar 采样器与基础分布式采样器链式结合
            Some(LlamaSampler::chain_simple([
                gs,
                LlamaSampler::dist(self.config.seed),
            ]))
        } else {
            None
        };

        while self.n_past < n_len {
            // Check for external cancellation
            if let Some(ref c) = self.cancel {
                if c.load(Ordering::Relaxed) {
                    tracing::info!("LLM generation cancelled by external signal");
                    break;
                }
            }

            // 采样
            let token = if let Some(ref mut gs) = grammar_sampler {
                let t = gs.sample(&self.ctx, logit_idx);
                t
            } else {
                let t = self.sampler.sample(&self.ctx, logit_idx);
                t
            };

            // 检查结束
            if self.model.is_eog_token(token) {
                break;
            }

            // Token -> 文本
            let output_bytes = self.model.token_to_piece_bytes(token, 512, true, None)?;
            let mut token_str = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_str, false);

            output.push_str(&token_str);
            self.stats.total_generated += 1;

            // 如果启用了本地解析（工具调用或思维链），则使用 parser 发送 delta
            if let Some(ref mut parser) = oai_parser {
                match parser.update(&token_str, true) {
                    Ok(deltas) => {
                        for delta in deltas {
                            if let Some(ref mut cb) = callback {
                                cb(&delta);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("OAI 解析增量失败: {:?}", e);
                        if let Some(ref mut cb) = callback {
                            cb(&token_str);
                        }
                    }
                }
            } else {
                // 流式回调原始文本
                if let Some(ref mut cb) = callback {
                    cb(&token_str);
                }
            }

            // 编码新生成的 token，更新 KV Cache
            batch.clear();
            batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;
            self.encoded_tokens.push(token);

            self.ctx.decode(&mut batch).context("生成解码失败")?;

            // 下一次采样时，logits 在新 batch 的位置 0
            logit_idx = 0;
        }

        // 结束时的最终解析
        if let Some(mut parser) = oai_parser {
            if let Ok(deltas) = parser.update("", false) {
                for delta in deltas {
                    if let Some(ref mut cb) = callback {
                        cb(&delta);
                    }
                }
            }
        }

        let duration = start_time.elapsed();
        tracing::info!(
            "本地 LLM 回复完成，长度: {} bytes, 耗时: {:?}, KV: {}/{}",
            output.len(),
            duration,
            self.n_past,
            self.ctx.n_ctx()
        );

        // 7. 添加 assistant 消息到历史
        self.messages.push(ChatMessage::assistant(&output));
        self.stats.turn_count += 1;
        self.stats.tokens_used = self.n_past;

        Ok(output)
    }

    /// 找出新 tokens 与已编码 tokens 的公共前缀长度
    fn find_common_prefix(&self, new_tokens: &[LlamaToken]) -> usize {
        let mut common = 0;
        for (old, new) in self.encoded_tokens.iter().zip(new_tokens.iter()) {
            if old == new {
                common += 1;
            } else {
                break;
            }
        }
        common
    }

    /// 构建完整的提示词
    fn build_prompt(&self) -> Result<ChatTemplateResult> {
        let tmpl = &self.chat_template;

        // 构建 OpenAI 兼容的 JSON 消息格式
        let messages_json: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "role": m.role.as_str(),
                    "content": m.content
                });
                if let Some(ref tc) = m.tool_calls {
                    obj["tool_calls"] = tc.clone();
                }
                if let Some(ref tid) = m.tool_call_id {
                    obj["tool_call_id"] = serde_json::Value::String(tid.clone());
                }
                obj
            })
            .collect();
        let messages_json_str = serde_json::to_string(&messages_json).context("序列化消息失败")?;

        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json_str,
            tools_json: self.tools_json.as_deref(),
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: None,
            chat_template_kwargs: None,
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: self.config.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: self.tools_json.is_some(),
        };

        match self.model.apply_chat_template_oaicompat(tmpl, &params) {
            Ok(result) => {
                tracing::debug!(
                    "Chat template applied (enable_thinking={}, has_tools={})",
                    self.config.enable_thinking,
                    self.tools_json.is_some()
                );
                Ok(result)
            }
            Err(e) => {
                tracing::warn!("应用 oaicompat 聊天模板失败: {:?}, 将尝试使用 chatml 兜底", e);
                // 如果原始模板失败（可能是由于 Jinja 兼容性问题），尝试使用 chatml 兜底
                let fallback_tmpl = LlamaChatTemplate::new("chatml").context("无法创建 chatml 兜底模板")?;
                match self.model.apply_chat_template_oaicompat(&fallback_tmpl, &params) {
                    Ok(result) => {
                        tracing::info!("使用 chatml 兜底模板成功");
                        Ok(result)
                    }
                    Err(e2) => {
                        anyhow::bail!("应用聊天模板失败 (原始错误: {:?}, 兜底错误: {:?})", e, e2);
                    }
                }
            }
        }
    }

    /// 添加消息到历史（不触发生成）
    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    /// 设置完整的对话历史
    pub fn set_messages(&mut self, messages: Vec<ChatMessage>) {
        self.messages = messages;
        // 重置编码状态以触发重新 build_prompt 和 diff
        self.n_past = 0;
        self.encoded_tokens.clear();
    }

    /// 获取对话历史
    pub fn history(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// 设置工具定义 JSON
    pub fn set_tools_json(&mut self, json: Option<String>) {
        self.tools_json = json;
    }

    /// 设置外部取消标志
    ///
    /// 生成循环会在每个 token 采样前检查此标志，
    /// 若为 true 则立即停止生成。
    pub fn set_cancel(&mut self, cancel: Option<Arc<AtomicBool>>) {
        self.cancel = cancel;
    }

    /// 获取对话轮次数
    pub fn turn_count(&self) -> usize {
        self.stats.turn_count
    }

    /// 获取统计信息
    pub fn stats(&self) -> &SessionStats {
        &self.stats
    }

    /// 获取剩余上下文容量
    pub fn remaining_ctx(&self) -> i32 {
        self.ctx.n_ctx() as i32 - self.n_past
    }

    /// 获取 KV Cache 复用率
    pub fn cache_reuse_rate(&self) -> f32 {
        let total = self.stats.total_prefilled + self.stats.cache_hits;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f32 / total as f32
        }
    }

    /// 清除对话历史（保留 system prompt）
    ///
    /// 注意：这会重置 KV Cache
    pub fn clear(&mut self) {
        let system = self
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .cloned();

        self.messages.clear();
        if let Some(s) = system {
            self.messages.push(s);
        }

        // 重置 KV Cache 状态
        self.n_past = 0;
        self.encoded_tokens.clear();
        self.stats.turn_count = 0;
        self.stats.tokens_used = 0;
    }

    /// 完全重置会话（包括 system prompt）
    pub fn reset(&mut self) {
        self.messages.clear();
        self.n_past = 0;
        self.encoded_tokens.clear();
        self.stats = SessionStats {
            tokens_used: 0,
            ctx_size: self.ctx.n_ctx(),
            turn_count: 0,
            total_generated: 0,
            total_prefilled: 0,
            cache_hits: 0,
        };

        if let Some(ref system_prompt) = self.config.system_prompt {
            self.messages
                .push(ChatMessage::system(system_prompt.clone()));
        }
    }

    /// 设置新的 system prompt（会清除历史）
    pub fn set_system_prompt(&mut self, prompt: impl Into<String>) {
        self.messages.clear();
        self.messages.push(ChatMessage::system(prompt));
        self.n_past = 0;
        self.encoded_tokens.clear();
    }

    /// 获取会话信息
    pub fn info(&self) -> String {
        format!(
            "KV: {}/{}, 轮次: {}, 生成: {}, 复用率: {:.1}%",
            self.n_past,
            self.ctx.n_ctx(),
            self.stats.turn_count,
            self.stats.total_generated,
            self.cache_reuse_rate() * 100.0
        )
    }

    /// 估算剩余可用轮次
    pub fn estimate_remaining_turns(&self) -> usize {
        let remaining = self.remaining_ctx();
        let avg_tokens_per_turn = if self.stats.turn_count > 0 {
            self.n_past / self.stats.turn_count as i32
        } else {
            200
        };

        if avg_tokens_per_turn > 0 {
            (remaining / avg_tokens_per_turn) as usize
        } else {
            0
        }
    }
}
