use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText};
use llama_cpp_2::openai::OpenAIChatTemplateParams;
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub model_path: PathBuf,
    pub mmproj_path: PathBuf,
    pub ctx_size: NonZeroU32,
    pub max_tokens: u32,
    pub n_threads: Option<i32>,
    pub seed: u32,
    pub enable_thinking: bool,
}

impl VisionConfig {
    pub fn new<P1: Into<PathBuf>, P2: Into<PathBuf>>(model_path: P1, mmproj_path: P2) -> Self {
        Self {
            model_path: model_path.into(),
            mmproj_path: mmproj_path.into(),
            ctx_size: NonZeroU32::new(4096).unwrap(),
            max_tokens: 1024,
            n_threads: None,
            seed: 1234,
            enable_thinking: true,
        }
    }

    pub fn with_ctx_size(mut self, size: u32) -> Self {
        if let Some(s) = NonZeroU32::new(size) {
            self.ctx_size = s;
        }
        self
    }

    pub fn with_enable_thinking(mut self, enable: bool) -> Self {
        self.enable_thinking = enable;
        self
    }
}

pub struct VisionLlmModel {
    backend: Arc<LlamaBackend>,
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    config: VisionConfig,
}

impl VisionLlmModel {
    pub fn load(config: VisionConfig) -> Result<Self> {
        let backend = Arc::new(LlamaBackend::init().context("Failed to init backend")?);
        Self::load_with_backend(backend, config)
    }

    pub fn load_with_backend(backend: Arc<LlamaBackend>, config: VisionConfig) -> Result<Self> {
        let mut model_params = LlamaModelParams::default();
        model_params = model_params.with_n_gpu_layers(1_000_000); // Use all layers on GPU

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
            config.mmproj_path.to_string_lossy().as_ref(),
            &model,
            &mtmd_params,
        )
        .with_context(|| format!("Failed to load mmproj: {:?}", config.mmproj_path))?;

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
            .apply_chat_template_oaicompat(&chat_template, &params)?;
        let formatted_prompt = result.prompt;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        let n_past = chunks.eval_chunks(&self.mtmd_ctx, &mut context, 0, 0, 1, true)?;

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
            .apply_chat_template_oaicompat(&chat_template, &params)?;
        let formatted_prompt = result.prompt;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: true,
            parse_special: true,
        };

        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;
        let n_past = chunks.eval_chunks(&self.mtmd_ctx, &mut context, 0, 0, 1, true)?;

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
