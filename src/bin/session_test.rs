//! LLM 会话测试程序
//!
//! 用法:
//! ```
//! cargo run -p navi-llm --bin session_test -- --model /path/to/model.gguf
//! ```

use std::io::{self, BufRead, Write};

use anyhow::Result;
use clap::Parser;
use navi_llm::{LlmConfig, LlmSessionFactory};

/// LLM 会话测试工具 - 多轮对话
#[derive(Parser, Debug)]
#[command(name = "session_test")]
#[command(about = "测试 LLM 多轮对话功能")]
struct Args {
    /// GGUF 模型文件路径
    #[arg(short, long)]
    model: String,

    /// 系统提示词 (System Prompt)
    #[arg(short, long)]
    system: Option<String>,

    /// 上下文大小
    #[arg(long, default_value = "4096")]
    ctx_size: u32,

    /// 最大生成 token 数
    #[arg(long, default_value = "512")]
    max_tokens: u32,

    /// 使用流式输出
    #[arg(long)]
    streaming: bool,

    /// 启用详细日志
    #[arg(short, long)]
    verbose: bool,

    /// 禁用思考模式 (Qwen3.5 <think> 标签)
    #[arg(long)]
    no_think: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 初始化 tracing
    let level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .init();

    println!("========================================");
    println!("    navi-llm 多轮对话测试");
    println!("========================================");
    println!();

    // 构建配置
    let mut config = LlmConfig::new(&args.model)
        .with_ctx_size(args.ctx_size)
        .with_max_tokens(args.max_tokens)
        .with_verbose(args.verbose)
        .with_enable_thinking(!args.no_think);

    if let Some(system) = &args.system {
        config = config.with_system_prompt(system);
    }

    println!("📂 模型: {}", args.model);
    if let Some(system) = &args.system {
        println!("🧠 System: {}", system);
    }
    println!("📊 上下文大小: {}", args.ctx_size);
    println!("🔢 最大生成 tokens: {}", args.max_tokens);
    println!();

    // 创建会话工厂
    println!("⏳ 正在加载模型...");
    let start = std::time::Instant::now();
    let factory = LlmSessionFactory::new(config)?;
    println!(
        "✅ 模型加载完成 (耗时: {:.2}s)",
        start.elapsed().as_secs_f32()
    );
    println!();

    // 创建会话
    let mut session = factory.create_session()?;

    println!("💬 开始对话 (输入 /exit 退出, /clear 清除历史, /info 查看状态)");
    println!("----------------------------------------");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // 读取用户输入
        print!("\n🧑 You: ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // 处理命令
        if input.starts_with('/') {
            match input {
                "/exit" | "/quit" => {
                    println!("👋 再见!");
                    break;
                }
                "/clear" => {
                    session.clear();
                    println!("🗑️ 对话历史已清除");
                    continue;
                }
                "/info" => {
                    println!("📊 {}", session.info());
                    println!("📜 历史消息数: {}", session.history().len());
                    println!("🔮 预估剩余轮次: ~{}", session.estimate_remaining_turns());
                    continue;
                }
                "/history" => {
                    println!("📜 对话历史:");
                    for (i, msg) in session.history().iter().enumerate() {
                        println!(
                            "  [{}] {:?}: {}",
                            i,
                            msg.role,
                            &msg.content[..msg.content.len().min(50)]
                        );
                    }
                    continue;
                }
                "/reset" => {
                    session.reset();
                    println!("🔄 会话已完全重置");
                    continue;
                }
                _ => {
                    println!("❓ 未知命令: {}", input);
                    println!("   可用命令: /exit, /clear, /info, /history, /reset");
                    continue;
                }
            }
        }

        // 生成回复
        print!("🤖 Assistant: ");
        stdout.flush()?;

        let start = std::time::Instant::now();

        let reply = if args.streaming {
            session.chat_streaming(input, |token| {
                print!("{}", token);
                stdout.flush().ok();
            })?
        } else {
            let reply = session.chat(input)?;
            print!("{}", reply);
            reply
        };

        println!();

        let elapsed = start.elapsed();
        let token_count = reply.chars().count();
        let stats = session.stats();
        println!(
            "   [~{} 字符, {:.2}s, KV {}/{}, 复用率 {:.1}%]",
            token_count,
            elapsed.as_secs_f32(),
            stats.tokens_used,
            stats.ctx_size,
            session.cache_reuse_rate() * 100.0
        );
    }

    Ok(())
}
