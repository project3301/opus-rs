//! Issue #48: stereo SILK and Hybrid code mid/side like libopus's
//! `silk_stereo_LR_to_MS`, checked per channel against libopus (the `opus`
//! dev-dependency).
//!
//! Each case encodes one stereo signal with opus-rs and with libopus at the
//! same settings, decodes both streams with libopus, and compares what comes
//! out against the input, channel by channel. Assertions are relative to
//! libopus's own encoder. The opus-rs decoder must agree with libopus on
//! opus-rs's packets.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;
const CH: usize = 2;
const SECONDS: usize = 2;
/// Best-lag search range for the input comparisons.
const LAG_MS: usize = 40;

/// A 140 Hz harmonic series with slow pitch drift and a syllable-rate
/// envelope, as in `issue_42_stereo_channels.rs`, mono.
fn voiced_mono(sr: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
            phase += 2.0 * PI * f0 / sr as f32;
            let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
            let s: f32 = (1..=8)
                .map(|h| 0.25 / h as f32 * (h as f32 * phase).sin())
                .sum();
            s * env * 0.5
        })
        .collect()
}

/// Amplitude-panned speech: L = 0.9·s, R = 0.3·s. The side is exactly 0.5·mid,
/// so the stereo predictor alone carries the image.
fn panned(sr: usize, n: usize) -> Vec<f32> {
    voiced_mono(sr, n)
        .iter()
        .flat_map(|&s| [0.9 * s, 0.3 * s])
        .collect()
}

fn channel(pcm: &[f32], c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(CH).copied().collect()
}

fn mid(pcm: &[f32]) -> Vec<f32> {
    pcm.chunks_exact(CH).map(|p| (p[0] + p[1]) / 2.0).collect()
}

/// Sums over `reference[start..start + len]` against `test` shifted by `lag`:
/// (Σr², Σr·t, Σ(r−t)²).
fn sums(reference: &[f32], test: &[f32], start: usize, len: usize, lag: isize) -> (f64, f64, f64) {
    let (mut rr, mut rt, mut err) = (0f64, 0f64, 0f64);
    for i in start..start + len {
        let j = i as isize + lag;
        if j < 0 || j as usize >= test.len() || i >= reference.len() {
            continue;
        }
        let (r, t) = (reference[i] as f64, test[j as usize] as f64);
        rr += r * r;
        rt += r * t;
        err += (r - t) * (r - t);
    }
    (rr, rt, err)
}

fn snr_db(reference: &[f32], test: &[f32], start: usize, len: usize, lag: isize) -> f64 {
    let (rr, _, err) = sums(reference, test, start, len, lag);
    10.0 * (rr / err.max(1e-20)).log10()
}

/// The lag in `-max_lag..=max_lag` at which `test` best matches `reference`.
fn best_lag(reference: &[f32], test: &[f32], start: usize, len: usize, max_lag: isize) -> isize {
    (-max_lag..=max_lag)
        .max_by(|&a, &b| {
            snr_db(reference, test, start, len, a)
                .total_cmp(&snr_db(reference, test, start, len, b))
        })
        .unwrap()
}

/// What one decoded stream looks like against the input.
#[derive(Debug)]
struct Channels {
    /// Least-squares gain of each decoded channel against the input mid.
    gain: [f64; 2],
    /// SNR of each decoded channel against the same input channel (dB).
    snr: [f64; 2],
}

/// Measures `decoded` against `input` (both interleaved stereo), at the lag
/// that best aligns the decoded mid with the input mid.
fn measure(input: &[f32], decoded: &[f32], sr: usize) -> Channels {
    let (in_mid, out_mid) = (mid(input), mid(decoded));
    let max_lag = (sr * LAG_MS / 1000) as isize;
    // Skip 100 ms of encoder start-up; stop short of the lag search range.
    let start = sr / 10;
    let len = in_mid.len().min(out_mid.len()) - start - max_lag as usize;
    let lag = best_lag(&in_mid, &out_mid, start, len, max_lag);
    let mut out = Channels {
        gain: [0.0; 2],
        snr: [0.0; 2],
    };
    for c in 0..CH {
        let (in_c, out_c) = (channel(input, c), channel(decoded, c));
        let (rr, rt, _) = sums(&in_mid, &out_c, start, len, lag);
        out.gain[c] = rt / rr;
        out.snr[c] = snr_db(&in_c, &out_c, start, len, lag);
    }
    out
}

/// Worst per-20 ms-window SNR of channel `c` of `test` against `reference`,
/// at the best lag within ±`max_lag` of each window.
fn worst_window_snr(reference: &[f32], test: &[f32], c: usize, sr: usize, skip: usize) -> f64 {
    let (r, t) = (channel(reference, c), channel(test, c));
    let (win, max_lag) = (sr / 50, 30isize);
    let n = r.len().min(t.len());
    let mut worst = f64::INFINITY;
    let mut start = skip;
    while start + win + max_lag as usize <= n {
        let lag = best_lag(&r, &t, start, win, max_lag);
        worst = worst.min(snr_db(&r, &t, start, win, lag));
        start += win;
    }
    worst
}

struct Case {
    name: &'static str,
    sr: usize,
    bitrate: i32,
    cbr: bool,
    bandwidth: CBw,
    signal: fn(usize, usize) -> Vec<f32>,
}

struct Measured {
    /// opus-rs's packets, decoded by libopus.
    rs: Channels,
    /// libopus's packets, decoded by libopus.
    c: Channels,
    /// Per channel, worst-window SNR of opus-rs's decode of opus-rs's packets
    /// against libopus's decode of the same packets.
    agreement: [f64; 2],
}

fn run(case: &Case) -> Measured {
    let sr = case.sr;
    let frame = sr / 50;
    let packets = SECONDS * 50;
    let input = (case.signal)(sr, frame * packets);

    let mut rs_enc = OpusEncoder::new(sr as i32, CH, Application::Voip).unwrap();
    rs_enc.bitrate_bps = case.bitrate;
    rs_enc.use_cbr = case.cbr;

    // libopus at opus-rs's settings, pinned to voice and to a stereo stream
    // (below ~18 kb/s it would otherwise switch to mono).
    let mut c_enc = CEnc::new(sr as u32, CCh::Stereo, CApp::Voip).unwrap();
    c_enc.set_bitrate(CBitrate::Bits(case.bitrate)).unwrap();
    c_enc.set_vbr(!case.cbr).unwrap();
    c_enc.set_complexity(rs_enc.complexity).unwrap();
    c_enc.set_bandwidth(case.bandwidth).unwrap();
    c_enc.set_signal(CSignal::Voice).unwrap();
    c_enc.set_force_channels(Some(CCh::Stereo)).unwrap();

    let mut c_dec_rs = CDec::new(sr as u32, CCh::Stereo).unwrap();
    let mut c_dec_c = CDec::new(sr as u32, CCh::Stereo).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, CH).unwrap();

    let (mut rs_by_c, mut c_by_c, mut rs_by_rs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pkt, mut buf) = (vec![0u8; 1500], vec![0f32; MAX_FRAME * CH]);
    for p in 0..packets {
        let pcm = &input[p * frame * CH..(p + 1) * frame * CH];

        let n = rs_enc
            .encode(pcm, frame, &mut pkt)
            .unwrap_or_else(|e| panic!("{}: packet {p}: opus-rs encode failed: {e}", case.name));
        let rs_pkt = &pkt[..n];
        assert_eq!(
            (rs_pkt[0] >> 2) & 1,
            1,
            "{}: packet {p}: TOC stereo bit",
            case.name
        );

        let got = c_dec_rs.decode_float(rs_pkt, &mut buf, false).unwrap();
        assert_eq!(got, frame);
        rs_by_c.extend_from_slice(&buf[..got * CH]);

        let got = rs_dec.decode(rs_pkt, MAX_FRAME, &mut buf).unwrap();
        assert_eq!(got, frame);
        rs_by_rs.extend_from_slice(&buf[..got * CH]);

        let n = c_enc.encode_float(pcm, &mut pkt).unwrap();
        let got = c_dec_c.decode_float(&pkt[..n], &mut buf, false).unwrap();
        assert_eq!(got, frame);
        c_by_c.extend_from_slice(&buf[..got * CH]);
    }

    let skip = sr / 10;
    Measured {
        rs: measure(&input, &rs_by_c, sr),
        c: measure(&input, &c_by_c, sr),
        agreement: [0, 1].map(|c| worst_window_snr(&rs_by_c, &rs_by_rs, c, sr, skip)),
    }
}

/// Decoder agreement on opus-rs's packets (issue_27's `MATCH_DB`).
const MATCH_DB: f64 = 30.0;

fn report(case: &Case, m: &Measured) {
    println!(
        "{:<26} opus-rs: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} dB",
        case.name, m.rs.gain[0], m.rs.gain[1], m.rs.snr[0], m.rs.snr[1]
    );
    println!(
        "{:<26} libopus: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} dB",
        "", m.c.gain[0], m.c.gain[1], m.c.snr[0], m.c.snr[1]
    );
    println!(
        "{:<26} opus-rs vs libopus decoder: L {:.1} R {:.1} dB",
        "", m.agreement[0], m.agreement[1]
    );
    for (c, name) in ["L", "R"].iter().enumerate() {
        assert!(
            m.agreement[c] > MATCH_DB,
            "{}: opus-rs and libopus decode opus-rs's {name} differently: {:.1} dB",
            case.name,
            m.agreement[c]
        );
    }
}

/// Amplitude-panned speech at 16 kb/s: libopus codes it as panned mono, the
/// mid with a predictor and no side. The L/R balance must survive. Coding the
/// mid with a zero predictor decoded both channels as the mid (+1.0·mid).
/// CBR, because SILK-only VBR doesn't follow the bitrate yet (#43).
#[test]
fn panned_speech_keeps_its_balance_at_16k() {
    let case = Case {
        name: "panned SILK WB 16k CBR",
        sr: 16000,
        bitrate: 16000,
        cbr: true,
        bandwidth: CBw::Wideband,
        signal: panned,
    };
    let m = run(&case);
    report(&case, &m);
    for (c, name) in ["L", "R"].iter().enumerate() {
        assert!(
            (m.rs.gain[c] - m.c.gain[c]).abs() <= 0.1,
            "{}: decoded {name} = {:+.2}·mid, libopus {:+.2}·mid",
            case.name,
            m.rs.gain[c],
            m.c.gain[c]
        );
        assert!(
            m.rs.snr[c] >= m.c.snr[c] - 1.0,
            "{}: {name} SNR {:+.1} dB, libopus {:+.1} dB",
            case.name,
            m.rs.snr[c],
            m.c.snr[c]
        );
    }
    // The input is 1.5·mid and 0.5·mid; the predictor fades toward mono at
    // this rate, but L stays well above R.
    assert!(
        m.rs.gain[0] > 1.5 * m.rs.gain[1],
        "{}: L {:+.2}·mid vs R {:+.2}·mid: the panning was lost",
        case.name,
        m.rs.gain[0],
        m.rs.gain[1]
    );
}
