//! Issue #38: SILK/hybrid output at 48 kHz used to lag libopus's pre-skip by
//! up to 17 samples, and low-bitrate Audio mangled the first ~200 ms.
//!
//! Root causes (against libopus 1.3.1):
//! 1. The encoder resampled the API-rate input with the stateless
//!    `silk_resampler_down2` + `down2_3` / `down_1_3` decimators and then
//!    compensated the delay inside `silk_encode` with a hard-coded
//!    internal-rate delay buffer — the wrong rate and the wrong values
//!    (10 samples at 16 kHz ≈ 30 at 48 kHz, versus libopus's
//!    `delay_matrix_enc[48][16] = 12` plus the down-FIR group delay).
//! 2. `silk_encode_prefill` was never invoked, so SILK restarts after CELT
//!    started from a cold encoder state.
//!
//! The encoder now runs the libopus `silk_resampler` (encoder direction:
//! `delay_matrix_enc` + private down-FIR) on the API-rate input, and prefills
//! the SILK encoder on CELT→SILK restarts (opus_encoder.c:1448-1455,
//! 1806-1826). The alignment below is measured through libopus's decoder
//! after the 312-sample (6.5 ms) pre-skip, per 100 ms window, exactly like
//! the issue's harness. Reference values measured for libopus's own encoder
//! on the same signal: Voip 32 kbps lags -5..-2 samples at corr ≥ 0.99 after
//! a -12 dB first window; Audio 24 kbps runs CELT-only there, so opus-rs's
//! hybrid start is compared on its own (it converges to lag 0, corr 1.00).

use opus::{Channels as CCh, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

/// 1.3 s logarithmic-ish chirp, RMS ≈ 0.28 like the issue's repro.
fn chirp(sr: usize, seconds: usize) -> Vec<f32> {
    let n = sr * seconds / 10;
    let mut input = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / sr as f64;
        let phase =
            2.0 * std::f64::consts::PI * (100.0 * t + 1950.0 / 1.3 * t * t);
        input.push((0.28 * 2.0f64.sqrt() * phase.sin()) as f32);
    }
    input
}

/// Encode the whole stream with opus-rs, decode with libopus, and return the
/// per-window (best lag, correlation) after the 312-sample pre-skip.
fn lag_profile(app: Application, bitrate: i32) -> Vec<(i64, f64)> {
    let sr = 48000usize;
    let input = chirp(sr, 13);
    let frame = 960usize;
    let mut enc = OpusEncoder::new(sr as i32, 1, app).unwrap();
    enc.bitrate_bps = bitrate;
    let mut dec = CDec::new(sr as u32, CCh::Mono).unwrap();

    let mut pcm = Vec::new();
    let mut pkt = vec![0u8; 1500];
    for fr in 0..input.len() / frame {
        let len = enc
            .encode(&input[fr * frame..(fr + 1) * frame], frame, &mut pkt)
            .unwrap();
        let mut buf = vec![0f32; frame * 2];
        let got = dec.decode_float(&pkt[..len], &mut buf, false).unwrap_or(0);
        pcm.extend_from_slice(&buf[..got]);
    }

    let pre_skip = 312usize;
    let win = sr / 10;
    let mut profile = Vec::new();
    let mut start = 0usize;
    while start + win < input.len() && start + pre_skip + win < pcm.len() {
        let mut best = (0i64, f64::MIN);
        for lag in -40i64..=40 {
            let mut xy = 0f64;
            let mut xx = 0f64;
            let mut yy = 0f64;
            for i in start..start + win {
                let x = input[i] as f64;
                let j = (start + pre_skip) as i64 + (i - start) as i64 + lag;
                if j < 0 || j as usize >= pcm.len() {
                    continue;
                }
                let y = pcm[j as usize] as f64;
                xy += x * y;
                xx += x * x;
                yy += y * y;
            }
            let corr = xy / (xx.sqrt() * yy.sqrt() + 1e-12);
            if corr > best.1 {
                best = (lag, corr);
            }
        }
        profile.push(best);
        start += win;
    }
    profile
}

#[test]
fn voip_48k_tracks_libopus_preskip() {
    let profile = lag_profile(Application::Voip, 32000);
    // Steady state (from the second window on): within a couple of samples of
    // the pre-skip at waveform fidelity. libopus's own encoder measures
    // -5..-2 with the same harness.
    for (w, &(lag, corr)) in profile.iter().enumerate().skip(1) {
        assert!(
            lag.abs() <= 5,
            "Voip window {w}: lag {lag} vs libopus's -5..-2; profile {profile:?}"
        );
        assert!(
            corr >= 0.98,
            "Voip window {w}: correlation {corr:.3}; profile {profile:?}"
        );
    }
}

#[test]
fn audio_24k_hybrid_startup_recovers() {
    let profile = lag_profile(Application::Audio, 24000);
    // The hybrid startup previously measured corr 0.56-0.60 for ~200 ms and
    // is now ≥ 0.75 everywhere; the lag is 0 from the third window on
    // (libopus runs CELT-only at this setting, so only the recovery quality
    // is comparable).
    for (w, &(lag, corr)) in profile.iter().enumerate() {
        assert!(corr >= 0.75, "Audio window {w}: correlation {corr:.3}");
        assert!(lag.abs() <= 2, "Audio window {w}: lag {lag}");
    }
}
