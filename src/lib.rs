//! # navi-llm
//!
//! 本地 LLM 推理库，基于 llama-cpp-rs 运行 GGUF 格式模型。
//!
//! ## 使用示例
//!
//! ```rust,ignore
//! use navi_llm::{LlmConfig, LlmModel};
//!
//! let config = LlmConfig::new("/path/to/model.gguf");
//! let model = LlmModel::load(config)?;
//! let result = model.complete("Hello, world!")?;
//! println!("{}", result);
//! ```
//!
//! ## 多轮对话示例
//!
//! ```rust,ignore
//! use navi_llm::{LlmConfig, LlmSessionFactory};
//!
//! let config = LlmConfig::new("/path/to/model.gguf")
//!     .with_system_prompt("你是一个有帮助的助手。");
//! let factory = LlmSessionFactory::new(config)?;
//! let mut session = factory.create_session()?;
//!
//! let reply1 = session.chat("你好")?;
//! let reply2 = session.chat("请继续")?;
//! ```

mod config;
mod model;
mod sampler;
mod session;
mod vision;

pub use config::{LlmConfig, SamplingMode};
pub use llama_cpp_2::llama_backend::LlamaBackend;
pub use model::LlmModel;
pub use session::{ChatMessage, LlmSessionFactory, ManagedSession, Role, SessionStats};
pub use vision::VisionLlmModel;
