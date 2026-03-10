use std::ffi::CString;
use std::sync::Arc;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::openai::OpenAIChatTemplateParams;

use crate::config::LlmConfig;
use crate::sampler;

pub struct VisionLlmModel {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    config: LlmConfig,
}

impl VisionLlmModel {
    pub fn load(config: LlmConfig) -> Result<Self> {
        let backend = Arc::new(LlamaBackend::init().context("Failed to init backend")?);
        Self::load_with_backend(backend, config)
    }

    pub fn load_with_backend(backend: Arc<LlamaBackend>, config: LlmConfig) -> Result<Self> {
        let mmproj_path = config
            .mmproj_path
            .as_ref()
            .context("VisionLlmModel 需要配置 mmproj_path")?;

        let mut model_params = LlamaModelParams::default();
        if let Some(n) = config.n_gpu_layers {
            model_params = model_params.with_n_gpu_layers(n);
        } else {
            // 视觉推理默认全部 offload
            model_params = model_params.with_n_gpu_layers(1_000_000);
        }

        let model = LlamaModel::load_from_file(
            &backend,
            config.model_path.to_string_lossy().as_ref(),
            &model_params,
        )
        .with_context(|| format!("Failed to load model: {:?}", config.model_path))?;

        let mtmd_params = MtmdContextParams {
            use_gpu: true,
            print_timings: false,
            n_threads: config.n_threads.unwrap_or(4),
            media_marker: CString::new(llama_cpp_2::mtmd::mtmd_default_marker().to_string())?,
        };

        let mtmd_ctx = MtmdContext::init_from_file(
            &mmproj_path.to_string_lossy(),
            &model,
            &mtmd_params,
        )
        .with_context(|| format!("Failed to load mmproj: {:?}", mmproj_path))?;

        Ok(Self {
            backend,
            model,
            mtmd_ctx,
            config,
        })
    }

    pub fn complete_with_image(&self, prompt: &str, image_path: &str) -> Result<String> {
        let n_threads = self.config.n_threads.unwrap_or(4);
        let ctx_params = LlamaContextParams::default()
            .with_n_threads(n_threads)
            .with_n_ctx(Some(self.config.ctx_size));

        let mut context = self.model.new_context(&self.backend, ctx_params)?;

        // Add marker if not present
        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let mut full_prompt = prompt.to_string();
        if !full_prompt.contains(&default_marker) {
            full_prompt.push_str(&default_marker);
        }

        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, image_path)?;
        let bitmap_refs = vec![&bitmap];

        let chat_template = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        let messages_json = serde_json::json!([{"role": "user", "content": full_prompt}]);
        let messages_json_str = serde_json::to_string(&messages_json)?;

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

        let result = self
            .model
            .apply_chat_template_oaicompat(&chat_template, &params)?;
        let formatted_prompt = result.prompt;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        let n_past = chunks.eval_chunks(&self.mtmd_ctx, &mut context, 0, 0, 1, true)?;

        let mut sampler = sampler::build_sampler(&self.config);
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

            let piece = self.model.token_to_piece(token, &mut decoder, true, None)?;
            output.push_str(&piece);

            batch.clear();
            batch.add(token, current_past, &[0], true)?;
            current_past += 1;

            context.decode(&mut batch)?;
        }

        Ok(output)
    }

    /// 执行图像分析流式回调 (从 JPEG/PNG bytes)
    pub fn complete_with_image_bytes_streaming<F>(
        &self,
        prompt: &str,
        image_bytes: &[u8],
        callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let bitmap = MtmdBitmap::from_buffer(&self.mtmd_ctx, image_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to decode image bytes: {:?}", e))?;
        self.complete_with_bitmap_streaming(prompt, bitmap, callback)
    }

    /// 执行图像分析流式回调
    pub fn complete_with_image_streaming<F>(
        &self,
        prompt: &str,
        image_path: &str,
        callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, image_path)
            .map_err(|e| anyhow::anyhow!("Failed to load image file: {:?}", e))?;
        self.complete_with_bitmap_streaming(prompt, bitmap, callback)
    }

    /// 内部：基于 bitmap 的流式推理
    fn complete_with_bitmap_streaming<F>(
        &self,
        prompt: &str,
        bitmap: MtmdBitmap,
        mut callback: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let n_threads = self.config.n_threads.unwrap_or(4);
        let ctx_params = LlamaContextParams::default()
            .with_n_threads(n_threads)
            .with_n_ctx(Some(self.config.ctx_size));

        let mut context = self.model.new_context(&self.backend, ctx_params)?;

        // Add marker if not present
        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let mut full_prompt = prompt.to_string();
        if !full_prompt.contains(&default_marker) {
            full_prompt.push_str(&default_marker);
        }

        let bitmap_refs = vec![&bitmap];

        let chat_template = self
            .model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        let messages_json = serde_json::json!([{"role": "user", "content": full_prompt}]);
        let messages_json_str = serde_json::to_string(&messages_json)?;

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

        let result = self
            .model
            .apply_chat_template_oaicompat(&chat_template, &params)?;
        let formatted_prompt = result.prompt;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        let n_past = chunks.eval_chunks(&self.mtmd_ctx, &mut context, 0, 0, 1, true)?;

        let mut sampler = sampler::build_sampler(&self.config);
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
