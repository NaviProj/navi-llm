//! LLM 测试程序
//!
//! 用法:
//! ```
//! cargo run -p navi-llm --bin llm_test -- --model /path/to/model.gguf --prompt "你好"
//! ```

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use navi_llm::{LlmConfig, LlmModel};

/// LLM 测试工具 - 加载 GGUF 模型并执行文本补全
#[derive(Parser, Debug)]
#[command(name = "llm_test")]
#[command(about = "测试 GGUF 模型的文本补全功能")]
struct Args {
    /// GGUF 模型文件路径
    #[arg(short, long)]
    model: String,

    /// 输入提示词
    #[arg(short, long)]
    prompt: String,

    /// 系统提示词 (System Prompt)
    #[arg(short, long)]
    system: Option<String>,

    /// 最大生成 token 数
    #[arg(long, default_value = "128")]
    max_tokens: u32,

    /// 上下文大小
    #[arg(long, default_value = "2048")]
    ctx_size: u32,

    /// 推理线程数
    #[arg(short, long)]
    threads: Option<i32>,

    /// 随机种子
    #[arg(long, default_value = "1234")]
    seed: u32,

    /// 启用详细日志
    #[arg(short, long)]
    verbose: bool,

    /// 使用流式输出
    #[arg(long)]
    streaming: bool,

    /// 仅测试 Jinja 模板加载与渲染，不进行 LLM 推理
    #[arg(long)]
    test_template: bool,

    /// 禁用思考模式 (Qwen3.5 <think> 标签)
    #[arg(long)]
    no_think: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化 tracing
    if args.verbose {
        tracing_subscriber::fmt::init();
    }

    println!("========================================");
    println!("       navi-llm 测试程序");
    println!("========================================");
    println!();

    // 构建配置
    let mut config = LlmConfig::new(&args.model)
        .with_ctx_size(args.ctx_size)
        .with_max_tokens(args.max_tokens)
        .with_seed(args.seed)
        .with_verbose(args.verbose)
        .with_enable_thinking(!args.no_think);

    if let Some(threads) = args.threads {
        config = config.with_threads(threads);
    }

    if let Some(system) = &args.system {
        config = config.with_system_prompt(system);
    }

    println!("📂 模型路径: {}", args.model);
    if let Some(system) = &args.system {
        println!("🧠 System: {}", system);
    }
    println!("📝 Prompt: {}", args.prompt);
    println!("🔢 最大 tokens: {}", args.max_tokens);
    println!("📊 上下文大小: {}", args.ctx_size);
    println!();

    // 加载模型
    println!("⏳ 正在加载模型...");
    let start = std::time::Instant::now();

    if args.test_template {
        // 如果只测试模板，我们直接调用 llama_cpp_2 API，以避开完整的推理流程
        use llama_cpp_2::llama_backend::LlamaBackend;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::{LlamaChatMessage, LlamaModel as NativeLlamaModel};

        println!("🛠️ 正在测试 Jinja 模板解析...");
        let backend = LlamaBackend::init()?;
        let model_params = LlamaModelParams::default();
        let native_model = NativeLlamaModel::load_from_file(&backend, &args.model, &model_params)?;

        let template = native_model
            .chat_template(None)
            .map_err(|e| anyhow::anyhow!("模型内无内置 Jinja 模板 (或解析失败): {:?}", e))?;
        println!("✅ 成功提取模型内置 Chat Template");
        // Convert LlamaChatTemplate to c string and print
        // Note: LlamaChatTemplate has a .0 which is a CString.
        // We can print it using .to_string_lossy().
        // Since we can't directly access .0 if it's private, we can just use debug format.
        println!("====== 原始模板内容 ======");
        println!("{:?}", template);
        println!("========================");

        let mut messages = Vec::new();
        if let Some(system) = &args.system {
            messages.push(LlamaChatMessage::new("system".to_string(), system.clone())?);
        }
        messages.push(LlamaChatMessage::new(
            "user".to_string(),
            args.prompt.clone(),
        )?);

        // 这里推荐直接使用 oaicompat 渲染函数，因为它能较好地处理复杂的 Jinja（如工具调用 `tojson`）
        // 以及 C++ 层的动态缓冲区分配（以防因为缓冲大小导致 FFI Error -1 报错）
        let result = native_model
            .apply_chat_template_with_tools_oaicompat(&template, &messages, None, None, true)?;

        println!("====== Jinja 模板渲染后的最终 Prompt ======");
        println!("{}", result.prompt);
        println!("========================================");
        println!("✅ 测试模板成功完成");
        return Ok(());
    }

    let model = LlmModel::load(config)?;
    println!(
        "✅ 模型加载完成 (耗时: {:.2}s)",
        start.elapsed().as_secs_f32()
    );
    println!();

    // 执行补全
    println!("🚀 开始生成:");
    println!("----------------------------------------");
    print!("{}", args.prompt);
    std::io::stdout().flush()?;

    let start = std::time::Instant::now();

    let result = if args.streaming {
        // 流式输出
        model.complete_streaming(&args.prompt, |token| {
            print!("{}", token);
            std::io::stdout().flush().ok();
        })?
    } else {
        // 一次性输出
        let result = model.complete(&args.prompt)?;
        print!("{}", result);
        result
    };

    println!();
    println!("----------------------------------------");
    println!();

    let elapsed = start.elapsed();
    let token_count = result.chars().count(); // 粗略估计
    println!(
        "📈 生成完成: ~{} 字符, 耗时 {:.2}s",
        token_count,
        elapsed.as_secs_f32()
    );

    Ok(())
}
