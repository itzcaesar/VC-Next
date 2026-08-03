//! Play a WAV fixture through the native CPAL/WASAPI path for route validation.

#[path = "../audio.rs"]
mod audio;
#[path = "../inference.rs"]
mod inference;
#[path = "../live_sidecar.rs"]
mod live_sidecar;
#[path = "../sidecar.rs"]
mod sidecar;

use std::env;

use audio::play_wav_fixture;

fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0].eq_ignore_ascii_case(name))
        .map(|pair| pair[1].clone())
}

fn usage() {
    eprintln!(
        "Usage: native-fixture-playback --input <wav> --output <device id or name> [--seconds N] [--ready-file <json>]"
    );
}

fn run(args: &[String]) -> Result<(), String> {
    let input = option(args, "--input").ok_or_else(|| "--input is required.".to_owned())?;
    let output = option(args, "--output").ok_or_else(|| "--output is required.".to_owned())?;
    let seconds = option(args, "--seconds")
        .unwrap_or_else(|| "30".to_owned())
        .parse::<f64>()
        .map_err(|_| "--seconds must be a number.".to_owned())?;
    let ready_file = option(args, "--ready-file");
    let result = play_wav_fixture(&input, &output, seconds, ready_file.as_deref())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result)
            .map_err(|error| format!("Could not encode fixture playback report: {error}"))?
    );
    Ok(())
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        usage();
        std::process::exit(2);
    }
    if let Err(error) = run(&args) {
        eprintln!("native-fixture-playback: {error}");
        std::process::exit(1);
    }
}
