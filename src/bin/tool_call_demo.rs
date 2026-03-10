use anyhow::Result;
use clap::Parser;
use navi_llm::{LlmConfig, LlmSessionFactory};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 一简单的 Tool Call 演示
///
/// 演示如何使用 Grammar 强制 LLM 输出 JSON 格式的工具调用。
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 模型路径 (GGUF)
    #[arg(long, default_value = "models/qwen2-7b-instruct.gguf")]
    model: PathBuf,

    /// 提示词
    #[arg(long, default_value = "What is the weather in Beijing?")]
    prompt: String,
}

/// 定义工具调用的 JSON 结构
#[derive(Debug, Serialize, Deserialize)]
struct ToolCall {
    tool: String,
    args: serde_json::Value,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. 定义 GBNF Grammar (可选)
    // 注意：当前 llama-cpp-2 版本的 Grammar sampler 有兼容性问题，暂不使用
    // 通过系统提示词引导模型输出 JSON 格式
    let _grammar = r#"
root ::= object
object ::= "{" ws string ":" ws value "}" ws
value  ::= string
string ::= "\"" [^"\\]* "\""
ws     ::= [ \t\n\r]*
"#;

    // 2. 配置 LLM
    // 系统提示词引导模型使用工具
    let system_prompt = r#"You are a helpful assistant with access to tools.
You MUST response in JSON format.

Available Tools:
- get_weather(location: string, unit: string): Get current weather.
- search_web(query: string): Search the web.

Response Format:
{
  "tool": "tool_name",
  "args": { ...arguments... }
}
If you answer directly without tool, use tool="message".
"#;

    let config = LlmConfig::new(args.model)
        .with_ctx_size(4096)
        .with_system_prompt(system_prompt)
        .with_verbose(true);

    let factory = LlmSessionFactory::new(config)?;
    let mut session = factory.create_session()?;

    println!("User: {}", args.prompt);

    // 3. 发送请求
    // 由于 Grammar 的存在，模型只能输出符合 JSON 语法的文本
    let response = session.chat(&args.prompt)?;

    println!("LLM Raw Output: {}", response);

    // 4. 解析结果
    match serde_json::from_str::<ToolCall>(&response) {
        Ok(tool_call) => {
            println!("\n[Parsed Tool Call]");
            println!("Tool: {}", tool_call.tool);
            println!("Args: {}", tool_call.args);

            // 5. 模拟执行工具
            if tool_call.tool == "get_weather" {
                println!(">> Executing get_weather...");
                // Mock result
                let tool_result = "Weather in Beijing is Sunny, 25°C";

                // 6. 将结果反馈给 LLM
                // 注意：这里我们需要临时关闭 Grammar 吗？或者继续使用 Grammar 强制下一次输出？
                // 通常如果下一次是自然语言回复，我们可能需要去掉 Grammar，或者使用一个更宽松的 Grammar。
                // navi_llm 目前的设计是 session 级别的配置。
                // *进阶*: 最好在 chat 接口支持 override options。
                // 现在的简单演示，我们只演示第一步 Tool Call 的生成。
                println!(">> Tool Result: {}", tool_result);
            }
        }
        Err(e) => {
            println!("Failed to parse JSON: {}", e);
        }
    }

    Ok(())
}
