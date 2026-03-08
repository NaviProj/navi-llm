use std::ffi::CString;
use std::io::{self, Write};
use std::num::NonZeroU32;
use std::path::Path;

use clap::Parser;
use encoding_rs::UTF_8;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

/// Command line parameters for the MTMD CLI application
#[derive(clap::Parser, Debug)]
#[command(name = "mtmd-cli")]
#[command(about = "Experimental CLI for multimodal llama.cpp")]
pub struct MtmdCliParams {
    /// Path to the model file
    #[arg(short = 'm', long = "model", value_name = "PATH")]
    pub model_path: String,
    /// Path to the multimodal projection file
    #[arg(long = "mmproj", value_name = "PATH")]
    pub mmproj_path: String,
    /// Path to image file(s)
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<String>,
    /// Path to audio file(s)
    #[arg(long = "audio", value_name = "PATH")]
    pub audio: Vec<String>,
    /// Text prompt to use as input to the model. May include media markers - else they will be added automatically.
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub prompt: String,
    /// Number of tokens to predict (-1 for unlimited)
    #[arg(
        short = 'n',
        long = "n-predict",
        value_name = "N",
        default_value = "-1"
    )]
    pub n_predict: i32,
    /// Number of threads
    #[arg(short = 't', long = "threads", value_name = "N", default_value = "4")]
    pub n_threads: i32,
    /// Number of tokens to process in a batch during eval chunks
    #[arg(long = "batch-size", value_name = "b", default_value = "1")]
    pub batch_size: i32,
    /// Maximum number of tokens in context
    #[arg(long = "n-tokens", value_name = "N", default_value = "4096")]
    pub n_tokens: NonZeroU32,
    /// Chat template to use, default template if not provided
    #[arg(long = "chat-template", value_name = "TEMPLATE")]
    pub chat_template: Option<String>,
    /// Disable GPU acceleration
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,
    /// Disable GPU offload for multimodal projection
    #[arg(long = "no-mmproj-offload")]
    pub no_mmproj_offload: bool,
    /// Media marker. If not provided, the default marker will be used.
    #[arg(long = "marker", value_name = "TEXT")]
    pub media_marker: Option<String>,
}

/// State of the MTMD CLI application.
#[allow(missing_debug_implementations)]
pub struct MtmdCliContext<'a> {
    pub mtmd_ctx: MtmdContext,
    pub batch: LlamaBatch<'a>,
    pub bitmaps: Vec<MtmdBitmap>,
    pub n_past: i32,
    pub chat_template: LlamaChatTemplate,
    pub chat: Vec<LlamaChatMessage>,
}

impl<'a> MtmdCliContext<'a> {
    pub fn new(
        params: &MtmdCliParams,
        model: &LlamaModel,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mtmd_params = MtmdContextParams {
            use_gpu: !params.no_gpu && !params.no_mmproj_offload,
            print_timings: true,
            n_threads: params.n_threads,
            media_marker: CString::new(
                params
                    .media_marker
                    .as_ref()
                    .unwrap_or(&llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .clone(),
            )?,
        };

        let mtmd_ctx = MtmdContext::init_from_file(&params.mmproj_path, model, &mtmd_params)?;

        let chat_template = model
            .chat_template(params.chat_template.as_deref())
            .map_err(|e| format!("Failed to get chat template: {e}"))?;

        let batch = LlamaBatch::new(params.n_tokens.get() as usize, 1);

        Ok(Self {
            mtmd_ctx,
            batch,
            chat: Vec::new(),
            bitmaps: Vec::new(),
            n_past: 0,
            chat_template,
        })
    }

    pub fn load_media(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, path)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    pub fn eval_message(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        msg: LlamaChatMessage,
        add_bos: bool,
        batch_size: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.chat.push(msg);

        let formatted_prompt = model.apply_chat_template(&self.chat_template, &self.chat, true)?;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: add_bos,
            parse_special: true,
        };

        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();

        if bitmap_refs.is_empty() {
            println!("No bitmaps provided, only tokenizing text");
        } else {
            println!("Tokenizing with {} bitmaps", bitmap_refs.len());
        }

        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;

        println!("Tokenization complete, {} chunks created", chunks.len());

        self.bitmaps.clear();

        self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, 0, 0, batch_size, true)?;
        Ok(())
    }

    pub fn generate_response(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut generated_tokens = Vec::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };
        let mut decoder = UTF_8.new_decoder();

        for _i in 0..max_predict {
            let token = sampler.sample(context, -1);
            generated_tokens.push(token);
            sampler.accept(token);

            if model.is_eog_token(token) {
                println!();
                break;
            }

            let piece = model.token_to_piece(token, &mut decoder, true, None)?;
            print!("{piece}");
            io::stdout().flush()?;

            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            context.decode(&mut self.batch)?;
        }

        Ok(())
    }
}

fn run_single_turn(
    ctx: &mut MtmdCliContext,
    model: &LlamaModel,
    context: &mut LlamaContext,
    sampler: &mut LlamaSampler,
    params: &MtmdCliParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut prompt = params.prompt.clone();
    let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
    let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
    if !prompt.contains(media_marker) {
        prompt.push_str(media_marker);
    }

    for image_path in &params.images {
        println!("Loading image: {image_path}");
        ctx.load_media(image_path)?;
    }
    for audio_path in &params.audio {
        ctx.load_media(audio_path)?;
    }

    let msg = LlamaChatMessage::new("user".to_string(), prompt)?;

    println!("Evaluating message: {msg:?}");

    ctx.eval_message(model, context, msg, true, params.batch_size)?;
    ctx.generate_response(model, context, sampler, params.n_predict)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = MtmdCliParams::parse();

    if !Path::new(&params.model_path).exists() {
        eprintln!("Error: Model file not found: {}", params.model_path);
        return Err("Model file not found".into());
    }

    if !Path::new(&params.mmproj_path).exists() {
        eprintln!(
            "Error: Multimodal projection file not found: {}",
            params.mmproj_path
        );
        return Err("Multimodal projection file not found".into());
    }

    println!("Loading model: {}", params.model_path);

    let backend = LlamaBackend::init()?;
    let mut model_params = LlamaModelParams::default();
    if !params.no_gpu {
        model_params = model_params.with_n_gpu_layers(1_000_000);
    }

    let model = LlamaModel::load_from_file(&backend, &params.model_path, &model_params)?;

    let context_params = LlamaContextParams::default()
        .with_n_threads(params.n_threads)
        .with_n_batch(params.batch_size.try_into()?)
        .with_n_ctx(Some(params.n_tokens));
    let mut context = model.new_context(&backend, context_params)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    println!("Model loaded successfully");
    println!("Loading mtmd projection: {}", params.mmproj_path);

    let mut ctx = MtmdCliContext::new(&params, &model)?;

    run_single_turn(&mut ctx, &model, &mut context, &mut sampler, &params)?;

    println!("\n");

    Ok(())
}
