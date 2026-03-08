use anyhow::Result;
use clap::Parser;
use navi_llm::{VisionConfig, VisionLlmModel};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(
        short,
        long,
        default_value = "/Users/trevorlink/Project/tiebao/navi/models/llm/Qwen3.5-4B-Q4_1.gguf"
    )]
    model: String,

    #[arg(
        long,
        default_value = "/Users/trevorlink/Project/tiebao/navi/models/llm/mmproj-BF16.gguf"
    )]
    mmproj: String,

    #[arg(
        short,
        long,
        default_value = "/Users/trevorlink/Project/tiebao/navi/design_assets/navi_landing_page_concept.png"
    )]
    image: String,

    #[arg(short, long, default_value = "What is in this image?")]
    prompt: String,

    /// 禁用思考模式 (Qwen3.5 <think> 标签)
    #[arg(long)]
    no_think: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("初始化视觉语言模型...");
    println!("模型路径: {}", args.model);
    println!("投影路径: {}", args.mmproj);

    let config = VisionConfig::new(&args.model, &args.mmproj)
        .with_ctx_size(4096)
        .with_enable_thinking(!args.no_think);

    let model = VisionLlmModel::load(config)?;

    println!("模型加载成功！");
    println!("测试图像: {}", args.image);
    println!("输入提示: {}", args.prompt);
    println!("\n--- 流式输出开始 ---");

    model.complete_with_image_streaming(&args.prompt, &args.image, |piece| {
        print!("{}", piece);
        io::stdout().flush().unwrap();
    })?;

    println!("\n--- 流式输出结束 ---");
    Ok(())
}
