use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use std::io::LineWriter;
use std::sync::Mutex;

pub static PROGRESS_BAR: Lazy<Mutex<Option<ProgressBar>>> = Lazy::new(|| Mutex::new(None));

pub fn update_progress(position: u64) {
    let pb = PROGRESS_BAR.lock().unwrap();
    if let Some(pb) = pb.as_ref() {
        pb.set_position(position);
    }
}

pub fn increment_progress(amount: u64) {
    let pb = PROGRESS_BAR.lock().unwrap();
    if let Some(pb) = pb.as_ref() {
        pb.inc(amount);
    }
}

pub fn create_progress_bar(total_size: u64) {
    finish_progress_bar();
    let pb = ProgressBar::new(total_size);
    pb.enable_steady_tick(100);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} {msg} ({eta_precise} remaining)")
        .progress_chars("=>-"));
    *PROGRESS_BAR.lock().unwrap() = Some(pb);
}

pub fn finish_progress_bar() {
    let mut pb = PROGRESS_BAR.lock().unwrap();
    if let Some(pb) = pb.as_ref() {
        pb.finish();
    }
    *pb = None;
}

pub struct PBWriter {}

impl PBWriter {
    pub fn new() -> Self {
        PBWriter {}
    }
}

impl std::io::Write for PBWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let pb = PROGRESS_BAR.lock().unwrap();
        if let Some(pb) = pb.as_ref() {
            pb.println(std::str::from_utf8(buf).unwrap());
            Ok(buf.len())
        } else {
            std::io::stderr().write(buf)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

pub fn setup_logging(log_filter: Option<String>) {
    tracing_subscriber::fmt::fmt()
        .with_writer(move || -> Box<dyn std::io::Write> {
            Box::new(LineWriter::new(PBWriter::new()))
        })
        .with_env_filter(log_filter.unwrap_or(std::env::var("RUST_LOG").unwrap_or("info".to_string())))
        .init();
}
