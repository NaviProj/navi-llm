//! LLM 模型封装

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

use crate::config::LlmConfig;
use crate::sampler;

/// LLM 模型包装器
pub struct LlmModel {
    backend: LlamaBackend,
    model: LlamaModel,
    config: LlmConfig,
}

impl LlmModel {
    /// 从配置加载模型
    pub fn load(config: LlmConfig) -> Result<Self> {
        // 初始化日志
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(config.verbose));

        tracing::info!("正在加载本地 LLM 模型: {:?}", config.model_path);
        let start = std::time::Instant::now();

        // 初始化后端
        let backend = LlamaBackend::init().context("无法初始化 llama 后端")?;

        // 配置模型参数
        let mut model_params = LlamaModelParams::default();
        if let Some(n) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n);
        }
        let model_params = model_params;

        // 加载模型
        let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
            .with_context(|| format!("无法加载模型: {:?}", config.model_path))?;

        let duration = start.elapsed();
        tracing::info!("本地 LLM 模型加载成功，耗时: {:?}", duration);

        Ok(Self {
            backend,
            model,
            config,
        })
    }

    /// 执行文本补全
    pub fn complete(&self, prompt: &str) -> Result<String> {
        // 配置上下文参数
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(self.config.ctx_size));

        if let Some(threads) = self.config.n_threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = self.config.n_threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // 创建上下文
        tracing::debug!(
            "正在创建新的 LLM 上下文 (ctx_size={})",
            self.config.ctx_size
        );
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("无法创建 llama 上下文")?;
        tracing::debug!("LLM 上下文创建成功");

        let start_time = std::time::Instant::now();
        tracing::info!("开始本地 LLM 补全 (non-streaming)");

        // 构建完整提示词
        let full_prompt = self.build_prompt(prompt)?;

        // 分词
        let tokens_list = self
            .model
            .str_to_token(&full_prompt, AddBos::Never)
            .with_context(|| format!("分词失败: {}", full_prompt))?;

        let n_len = tokens_list.len() as i32 + self.config.max_tokens as i32;

        // 检查 KV 缓存大小
        let n_ctx = ctx.n_ctx() as i32;
        if n_len > n_ctx {
            anyhow::bail!("需要的 KV 缓存大小 ({}) 超过上下文大小 ({})", n_len, n_ctx);
        }

        // 创建 batch
        let mut batch = LlamaBatch::new(512, 1);

        // 添加 prompt tokens
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.iter()) {
            let is_last = i == last_index;
            batch.add(*token, i, &[0], is_last)?;
        }

        // 解码 prompt
        ctx.decode(&mut batch).context("prompt 解码失败")?;

        // 创建采样器
        let mut sampler = sampler::build_sampler(&self.config);

        // 生成循环
        let mut n_cur = batch.n_tokens();
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= n_len {
            // 采样下一个 token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // 检查是否结束
            if self.model.is_eog_token(token) {
                break;
            }

            // 转换 token 为文本
            let output_bytes = self.model.token_to_piece_bytes(token, 512, true, None)?;
            let mut token_str = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_str, false);
            output.push_str(&token_str);

            // 准备下一轮
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("解码失败")?;
        }

        tracing::info!(
            "本地 LLM 补全完成 (non-streaming)，耗时: {:?}",
            start_time.elapsed()
        );

        Ok(output)
    }

    /// 执行文本补全（流式回调）
    pub fn complete_streaming<F>(&self, prompt: &str, mut callback: F) -> Result<String>
    where
        F: FnMut(&str),
    {
        // 配置上下文参数
        let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(self.config.ctx_size));

        if let Some(threads) = self.config.n_threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = self.config.n_threads_batch {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // 创建上下文
        tracing::debug!(
            "正在创建新的 LLM 上下文 (ctx_size={})",
            self.config.ctx_size
        );
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .context("无法创建 llama 上下文")?;
        tracing::debug!("LLM 上下文创建成功");

        let start_time = std::time::Instant::now();
        tracing::info!("开始本地 LLM 补全 (streaming)");

        // 构建完整提示词
        let full_prompt = self.build_prompt(prompt)?;

        // 分词
        let tokens_list = self
            .model
            .str_to_token(&full_prompt, AddBos::Never)
            .with_context(|| format!("分词失败: {}", full_prompt))?;

        let n_len = tokens_list.len() as i32 + self.config.max_tokens as i32;

        // 检查 KV 缓存大小
        let n_ctx = ctx.n_ctx() as i32;
        if n_len > n_ctx {
            anyhow::bail!("需要的 KV 缓存大小 ({}) 超过上下文大小 ({})", n_len, n_ctx);
        }

        // 创建 batch
        let mut batch = LlamaBatch::new(512, 1);

        // 添加 prompt tokens
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.iter()) {
            let is_last = i == last_index;
            batch.add(*token, i, &[0], is_last)?;
        }

        // 解码 prompt
        ctx.decode(&mut batch).context("prompt 解码失败")?;

        // 创建采样器
        let mut sampler = sampler::build_sampler(&self.config);

        // 生成循环
        let mut n_cur = batch.n_tokens();
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= n_len {
            // 采样下一个 token
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // 检查是否结束
            if self.model.is_eog_token(token) {
                break;
            }

            // 转换 token 为文本
            let output_bytes = self.model.token_to_piece_bytes(token, 512, true, None)?;
            let mut token_str = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_str, false);

            // 流式回调
            callback(&token_str);
            output.push_str(&token_str);

            // 准备下一轮
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            ctx.decode(&mut batch).context("解码失败")?;
        }

        tracing::info!(
            "本地 LLM 补全完成 (streaming)，耗时: {:?}",
            start_time.elapsed()
        );

        Ok(output)
    }

    /// 获取模型信息
    pub fn model_info(&self) -> String {
        format!(
            "模型路径: {:?}, 上下文大小: {}, 最大 tokens: {}",
            self.config.model_path, self.config.ctx_size, self.config.max_tokens
        )
    }

    /// 构建带模板的提示词
    fn build_prompt(&self, user_prompt: &str) -> Result<String> {
        use llama_cpp_2::model::LlamaChatTemplate;
        let tmpl = self.model.chat_template(None).unwrap_or_else(|_| {
            LlamaChatTemplate::new("chatml").expect("valid default template")
        });

        let mut messages_json = Vec::new();

        if let Some(system_prompt) = &self.config.system_prompt {
            messages_json.push(serde_json::json!({
                "role": "system",
                "content": system_prompt
            }));
        }

        messages_json.push(serde_json::json!({
            "role": "user",
            "content": user_prompt
        }));

        let messages_json_str = serde_json::to_string(&messages_json).context("序列化消息失败")?;

        let thinking_kwargs = serde_json::json!({
            "enable_thinking": self.config.enable_thinking
        }).to_string();
        let params = OpenAIChatTemplateParams {
            messages_json: &messages_json_str,
            tools_json: None,
            tool_choice: None,
            json_schema: None,
            grammar: None,
            reasoning_format: if self.config.enable_thinking { Some("deepseek") } else { None },
            chat_template_kwargs: Some(&thinking_kwargs),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: self.config.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: false,
        };

        match self.model.apply_chat_template_oaicompat(&tmpl, &params) {
            Ok(result) => {
                tracing::debug!(
                    "Chat template applied (enable_thinking={}, thinking_forced_open={})\n--- rendered prompt ---\n{}\n--- end ---",
                    self.config.enable_thinking,
                    result.thinking_forced_open,
                    result.prompt,
                );
                Ok(result.prompt)
            }
            Err(e) => {
                tracing::warn!("应用聊天模板失败: {:?}, 将尝试使用 chatml 兜底", e);
                let fallback_tmpl = LlamaChatTemplate::new("chatml").context("无法创建 chatml 兜底模板")?;
                match self.model.apply_chat_template_oaicompat(&fallback_tmpl, &params) {
                    Ok(result) => {
                        tracing::info!("使用 chatml 兜底模板成功");
                        Ok(result.prompt)
                    }
                    Err(_) => {
                        tracing::warn!("兜底模板也失败，使用原始输入");
                        Ok(user_prompt.to_string())
                    }
                }
            }
        }
    }
}
