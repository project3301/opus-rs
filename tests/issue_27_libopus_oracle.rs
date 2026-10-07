//! Issue #27 checked against libopus (the `opus` dev-dependency).
//!
//! `issue_15_27_repro.rs` covers #27 with opus-rs on both ends of the wire,
//! which cannot see a bug the encoder and decoder share (the original part 2
//! desync was found with the C decoder). These tests put libopus on one side:
//!
//! - libopus encodes 40/60 ms SILK packets and opus-rs decodes them at 48 kHz
//!   with a 120 ms output buffer — what a Discord voice receiver does. Every
//!   20 ms window must match libopus's own decode of the same packets, so a
//!   silent or garbled 2nd/3rd SILK frame fails on its own window.
//! - opus-rs encodes 40 ms stereo SILK, and libopus decodes it: each window
//!   must track the input as well as plain 20 ms packets do.
//! - opus-rs encodes SILK with in-band FEC (issue #36), and libopus decodes
//!   it normally and with `decode_fec`. Judged against libopus's own encoder
//!   at the same settings: FEC may cost the normal decode no more, the LBRR
//!   section must be sent about as often and recover about as well, and the
//!   recovered frames must play at the input's level, at CBR and VBR, 20 to
//!   60 ms.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

/// Voiced-speech-like test signal: a 140 Hz harmonic series with slow pitch
/// drift and a syllable-rate envelope. The right channel is a scaled, phase
/// shifted copy so the stereo side signal is not trivially zero.
fn voiced(sr: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * ch);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut l = 0.0;
        let mut r = 0.0;
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        if ch == 2 {
            out.push(r * env * 0.4);
        }
    }
    out
}

fn channel(pcm: &[f32], ch: usize, c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(ch).copied().collect()
}

/// SNR of `test` against `reference` (in dB), at the best lag in
/// `-max_lag..=max_lag`, over `reference[start..start + len]`.
fn best_lag_snr(reference: &[f32], test: &[f32], start: usize, len: usize, max_lag: isize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let (mut sig, mut err) = (0f64, 0f64);
        for i in start..start + len {
            let j = i as isize + lag;
            if j < 0 || j as usize >= test.len() || i >= reference.len() {
                continue;
            }
            let r = reference[i] as f64;
            let e = r - test[j as usize] as f64;
            sig += r * r;
            err += e * e;
        }
        if sig > 0.0 {
            best = best.max(10.0 * (sig / err.max(1e-20)).log10());
        }
    }
    best
}

/// Per-window SNR (worst channel) of `test` against `reference`, both
/// interleaved with `ch` channels, over consecutive `win`-sample windows
/// after skipping `skip` samples of warm-up.
fn window_snrs(
    reference: &[f32],
    test: &[f32],
    ch: usize,
    win: usize,
    skip: usize,
    max_lag: isize,
) -> Vec<f64> {
    let refs: Vec<Vec<f32>> = (0..ch).map(|c| channel(reference, ch, c)).collect();
    let tests: Vec<Vec<f32>> = (0..ch).map(|c| channel(test, ch, c)).collect();
    let n = refs[0].len().min(tests[0].len());
    let mut out = Vec::new();
    let mut start = skip;
    while start + win + max_lag as usize <= n {
        let worst = (0..ch)
            .map(|c| best_lag_snr(&refs[c], &tests[c], start, win, max_lag))
            .fold(f64::INFINITY, f64::min);
        out.push(worst);
        start += win;
    }
    out
}

/// TOC check: SILK-only (configs 0..=11), `ch` channels, and `ms` of audio
/// in total. Returns (SILK frames per Opus frame, Opus frames in the packet):
/// a 60 ms SILK frame carries 3 SILK frames inside one Opus frame.
fn assert_silk_toc(pkt: &[u8], ch: usize, ms: usize) -> (usize, usize) {
    let toc = pkt[0];
    let config = toc >> 3;
    assert!(config <= 11, "expected SILK-only TOC, got config {config}");
    let stereo = (toc >> 2) & 1 == 1;
    assert_eq!(stereo, ch == 2, "TOC stereo bit");
    let dur = [10, 20, 40, 60][(config & 3) as usize];
    let count = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => (pkt[1] & 0x3f) as usize,
    };
    assert_eq!(
        dur * count,
        ms,
        "TOC duration (config {config}, {count} frames)"
    );
    ((dur / 20).max(1), count)
}

const LAG_MS: usize = 40;

/// Mid signal (L+R)/2 of interleaved `ch`-channel audio (identity for mono).
fn mid(pcm: &[f32], ch: usize) -> Vec<f32> {
    pcm.chunks_exact(ch)
        .map(|p| p.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Part 3: libopus-encoded multi-frame SILK -> opus-rs decoder at 48 kHz
// ---------------------------------------------------------------------------

/// Encode ~1 s with libopus at `ms` frames, decode each packet with libopus
/// and with opus-rs (48 kHz, 120 ms output buffer), and return opus-rs's
/// per-20 ms-window SNR against libopus.
fn c_enc_rust_dec(ch: usize, ms: usize) -> Vec<f64> {
    let sr = 48000;
    let frame = sr * ms / 1000;
    let packets = 1000 / ms;
    let input = voiced(sr, ch, frame * packets);

    let mut enc = CEnc::new(sr as u32, c_channels(ch), CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(24000)).unwrap();
    enc.set_bandwidth(CBw::Wideband).unwrap();
    // A steady harmonic tone can read as music and pull libopus into CELT.
    enc.set_signal(CSignal::Voice).unwrap();
    let mut c_dec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, ch).unwrap();

    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut c_buf = vec![0f32; MAX_FRAME * ch];
    let mut rs_buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let n = enc
            .encode_float(&input[p * frame * ch..(p + 1) * frame * ch], &mut pkt)
            .unwrap();
        assert_silk_toc(&pkt[..n], ch, ms);

        let c_n = c_dec.decode_float(&pkt[..n], &mut c_buf, false).unwrap();
        let rs_n = rs_dec
            .decode(&pkt[..n], MAX_FRAME, &mut rs_buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(c_n, frame, "libopus decoded length");
        assert_eq!(rs_n, frame, "packet {p}: opus-rs decoded length");
        c_pcm.extend_from_slice(&c_buf[..c_n * ch]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n * ch]);
    }
    // Skip the first packet (encoder/decoder warm-up); windows are 20 ms.
    window_snrs(&c_pcm, &rs_pcm, ch, 960, frame, 30)
}

fn check_matches_libopus(ch: usize, ms: usize) {
    let snrs = c_enc_rust_dec(ch, ms);
    let worst = min(&snrs);
    println!(
        "C enc {ms} ms {ch}ch -> opus-rs dec @48k: {} windows, worst {worst:.1} dB",
        snrs.len()
    );
    assert!(
        worst > MATCH_DB,
        "{ms} ms {ch}ch: opus-rs output diverges from libopus; per-window SNR: {snrs:.1?}"
    );
}

/// How closely opus-rs must match libopus when both decode the same packets.
/// Set with margin under the 20 ms baseline measured by
/// `libopus_20ms_baseline_matches` below. A dropped or garbled SILK frame
/// scores around 0 dB, far below this.
const MATCH_DB: f64 = 30.0;

#[test]
fn libopus_20ms_baseline_matches() {
    check_matches_libopus(1, 20);
    check_matches_libopus(2, 20);
}

#[test]
fn libopus_40ms_mono_decodes_every_silk_frame() {
    check_matches_libopus(1, 40);
}

#[test]
fn libopus_60ms_mono_decodes_every_silk_frame() {
    check_matches_libopus(1, 60);
}

#[test]
fn libopus_40ms_stereo_decodes_every_silk_frame() {
    check_matches_libopus(2, 40);
}

#[test]
fn libopus_60ms_stereo_decodes_every_silk_frame() {
    check_matches_libopus(2, 60);
}

// ---------------------------------------------------------------------------
// Parts 1 and 2: opus-rs-encoded SILK -> libopus decoder
// ---------------------------------------------------------------------------

/// Which encoder produces the stream. libopus is the yardstick for what in-band
/// FEC should cost and recover at the same settings.
#[derive(Clone, Copy, PartialEq)]
enum Enc {
    Rust,
    Libopus,
}

/// Per-window measurements of one encoded stream, decoded by libopus.
struct Encoded {
    /// Normal decode against the input (mid channel).
    normal: Vec<f64>,
    /// libopus `decode_fec` output (the previous packet recovered from LBRR)
    /// against the input; empty without FEC.
    lbrr: Vec<f64>,
    /// Level of each normal-decode window relative to the input (dB).
    normal_level: Vec<f64>,
    /// Level of each `decode_fec` window relative to the input (dB); empty
    /// without FEC.
    lbrr_level: Vec<f64>,
    /// opus-rs's own decode of the same packets against libopus's; empty for
    /// libopus-encoded streams.
    agreement: Vec<f64>,
    /// Packets whose header sets the (mid) LBRR flag.
    lbrr_packets: usize,
    packets: usize,
}

/// libopus settings matching the opus-rs encoder in `encode_stream`.
fn libopus_encoder(ch: usize, fec: bool, cbr: bool) -> CEnc {
    let mut enc = CEnc::new(16000, c_channels(ch), CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(32000)).unwrap();
    enc.set_bandwidth(CBw::Wideband).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_vbr(!cbr).unwrap();
    if fec {
        enc.set_inband_fec(true).unwrap();
        enc.set_packet_loss_perc(40).unwrap();
    }
    enc
}

/// The (mid-channel) LBRR flag of a single-Opus-frame SILK-only packet. The
/// SILK header starts with one VAD flag per 20 ms SILK frame and then the LBRR
/// flag, in the top bits of the first frame byte. CBR packets are padded with
/// code 3 (RFC 6716 §3.2.5), so skip its frame-count byte and padding length.
fn has_lbrr(pkt: &[u8], ms: usize) -> bool {
    let data = match pkt[0] & 3 {
        0 => 1,
        3 => {
            assert_eq!(pkt[1] & 0x3f, 1, "expected one Opus frame");
            let mut i = 2;
            if pkt[1] & 0x40 != 0 {
                while pkt[i] == 255 {
                    i += 1;
                }
                i += 1;
            }
            i
        }
        code => panic!("unexpected packet code {code}"),
    };
    let silk_frames = (ms / 20).max(1);
    (pkt[data] >> (7 - silk_frames)) & 1 == 1
}

/// Per-window RMS level of `test` relative to `reference` (dB), both mono and
/// sample-aligned, over consecutive `win`-sample windows after `skip`. A
/// window whose reference is silent reads NaN, so indices stay positional.
fn window_levels(reference: &[f32], test: &[f32], win: usize, skip: usize) -> Vec<f64> {
    let rms =
        |x: &[f32]| (x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64).sqrt();
    let n = reference.len().min(test.len());
    let mut out = Vec::new();
    let mut start = skip;
    while start + win <= n {
        let r = rms(&reference[start..start + win]);
        out.push(if r > 1e-3 {
            20.0 * (rms(&test[start..start + win]) / r).log10()
        } else {
            f64::NAN
        });
        start += win;
    }
    out
}

/// Per-window level of the decoded `test` relative to the `input` it codes
/// (dB, NaN where the input is silent), after shifting `test` back by the
/// codec delay, found as the lag that best aligns the whole streams.
fn input_levels(input: &[f32], test: &[f32], win: usize, skip: usize) -> Vec<f64> {
    let n = input.len().min(test.len());
    let max_lag = 16 * LAG_MS;
    let lag = (0..max_lag)
        .max_by(|&a, &b| {
            let corr = |l: usize| -> f64 {
                (skip..n - max_lag)
                    .map(|i| input[i] as f64 * test[i + l] as f64)
                    .sum()
            };
            corr(a).partial_cmp(&corr(b)).unwrap()
        })
        .unwrap();
    window_levels(&input[..n - lag], &test[lag..n], win, skip)
}

/// Encode `input` (16 kHz, `ch` channels) in `ms` packets with `enc`, decode
/// it with libopus (and, for opus-rs streams, with opus-rs). With `fec`, also
/// decode every packet through a second libopus decoder with `fec = true`.
fn encode_stream(enc: Enc, input: &[f32], ch: usize, ms: usize, fec: bool, cbr: bool) -> Encoded {
    let sr = 16000;
    let frame = sr * ms / 1000;
    let packets = input.len() / (frame * ch);

    let mut rs_enc = OpusEncoder::new(sr as i32, ch, Application::Voip).unwrap();
    rs_enc.bitrate_bps = 32000;
    rs_enc.use_cbr = cbr;
    if fec {
        rs_enc.use_inband_fec = true;
        rs_enc.packet_loss_perc = 40;
    }
    let mut c_enc = libopus_encoder(ch, fec, cbr);
    let mut c_dec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut c_fec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, ch).unwrap();

    let (mut pcm, mut fec_pcm, mut rs_pcm) = (Vec::new(), Vec::new(), Vec::new());
    let mut lbrr_packets = 0;
    let mut pkt = vec![0u8; 1500];
    let mut buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let x = &input[p * frame * ch..(p + 1) * frame * ch];
        let n = match enc {
            Enc::Rust => rs_enc
                .encode(x, frame, &mut pkt)
                .unwrap_or_else(|e| panic!("packet {p}: opus-rs encode failed: {e}")),
            Enc::Libopus => c_enc.encode_float(x, &mut pkt).unwrap(),
        };
        assert_silk_toc(&pkt[..n], ch, ms);
        if has_lbrr(&pkt[..n], ms) {
            lbrr_packets += 1;
        }

        let got = c_dec
            .decode_float(&pkt[..n], &mut buf, false)
            .unwrap_or_else(|e| panic!("packet {p}: libopus rejected the packet: {e}"));
        assert_eq!(got, frame);
        pcm.extend_from_slice(&buf[..got * ch]);

        if enc == Enc::Rust {
            let got = rs_dec.decode(&pkt[..n], MAX_FRAME, &mut buf).unwrap();
            assert_eq!(got, frame);
            rs_pcm.extend_from_slice(&buf[..got * ch]);
        }

        if fec && p > 0 {
            // Recovers packet p-1 from packet p's LBRR data.
            let got = c_fec
                .decode_float(&pkt[..n], &mut buf[..frame * ch], true)
                .unwrap_or_else(|e| panic!("packet {p}: libopus FEC decode failed: {e}"));
            assert_eq!(got, frame);
            fec_pcm.extend_from_slice(&buf[..got * ch]);
        }
    }
    let win = sr / 50;
    let max_lag = (sr * LAG_MS / 1000) as isize;
    let reference = mid(input, ch);
    Encoded {
        normal: window_snrs(&reference, &mid(&pcm, ch), 1, win, frame, max_lag),
        // fec_pcm[k] is packet k's audio (recovered from packet k+1).
        lbrr: if fec {
            window_snrs(&reference, &mid(&fec_pcm, ch), 1, win, frame, max_lag)
        } else {
            Vec::new()
        },
        normal_level: input_levels(&reference, &mid(&pcm, ch), win, frame),
        lbrr_level: if fec {
            input_levels(&reference, &mid(&fec_pcm, ch), win, frame)
        } else {
            Vec::new()
        },
        agreement: if enc == Enc::Rust {
            window_snrs(&pcm, &rs_pcm, ch, win, frame, 30)
        } else {
            Vec::new()
        },
        lbrr_packets,
        packets,
    }
}

/// ~1 s of the voiced test signal at 16 kHz, a whole number of `ms` packets.
fn voiced_16k(ch: usize, ms: usize) -> Vec<f32> {
    let frame = 16 * ms;
    voiced(16000, ch, frame * (1000 / ms))
}

fn rust_enc_c_dec(ch: usize, ms: usize, fec: bool, cbr: bool) -> Encoded {
    encode_stream(Enc::Rust, &voiced_16k(ch, ms), ch, ms, fec, cbr)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Median of the non-NaN values.
fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Floor for opus-rs-encoded audio decoded by libopus, against the input.
/// SILK is not a waveform-matching codec, so this sits well below the
/// codec-agreement bar: clean 20 ms packets score 5-8 dB here; a desynced
/// frame scores around 0 dB or below.
const INPUT_FLOOR_DB: f64 = 3.0;

/// Part 2a: a 40 ms stereo packet must decode in libopus as well as 20 ms
/// packets do (the stereo header used to be written for frame 0 only).
#[test]
fn opus_rs_40ms_stereo_decodes_in_libopus() {
    let base = rust_enc_c_dec(2, 20, false, true);
    let multi = rust_enc_c_dec(2, 40, false, true);
    let (base_worst, worst) = (min(&base.normal), min(&multi.normal));
    println!(
        "opus-rs enc 40 ms stereo -> libopus dec: worst {worst:.1} dB \
         (20 ms baseline {base_worst:.1} dB); opus-rs vs libopus {:.1} dB",
        min(&multi.agreement)
    );
    assert!(
        base_worst > INPUT_FLOOR_DB,
        "20 ms baseline too weak: {:.1?}",
        base.normal
    );
    assert!(
        worst > INPUT_FLOOR_DB && worst > base_worst - 3.0,
        "40 ms stereo: a SILK frame desynced in libopus; per-window SNR: {:.1?}",
        multi.normal
    );
    assert!(
        min(&multi.agreement) > MATCH_DB,
        "decoders disagree: {:.1?}",
        multi.agreement
    );
}

// ---------------------------------------------------------------------------
// Part 2b / issue #36: in-band FEC, judged against libopus's own encoder
// ---------------------------------------------------------------------------

/// How much worse than libopus's own encoder opus-rs may do, in dB, on the
/// two things FEC trades: the normal decode it costs, and the LBRR recovery
/// it buys. Comparing against libopus at the same settings cancels what the
/// two encoders do differently regardless of FEC (opus-rs codes mid-only
/// stereo, #42; SILK-only VBR ignores the bitrate, #43) and the libopus
/// version the `opus` crate links (1.3 bundled, or the system's 1.5+).
/// Measured against 1.5.2: opus-rs's LBRR recovers 0.1 dB less to 1.1 dB
/// more than libopus's, and FEC costs it no more; at CBR, before issue #36,
/// opus-rs sent no LBRR at all and recovered 5.6 dB less.
const FEC_MARGIN_DB: f64 = 0.5;

/// LBRR recovery floor, worst window. Even libopus's encoder leaves frames
/// below its speech gate without LBRR, and libopus conceals those at around
/// 0 dB; a desynced LBRR frame scores well below.
const LBRR_FLOOR_DB: f64 = -1.0;

/// Packets libopus's encoder protects that opus-rs may leave unprotected
/// (measured: at most 1).
const LBRR_PACKETS_SLACK: usize = 2;

/// Median level of the recovered frames against the input (dB). libopus's
/// own LBRR plays -3.6 to -1.4 dB, opus-rs's -2.1 to -0.5 dB; the old copy,
/// which reused the main frame's pulses at a raised gain, played about
/// +2.5 dB (issue #36).
const LBRR_LEVEL_DB: (f64, f64) = (-4.0, 1.0);

/// How far the normal decode's level may fall below libopus's encoder's with
/// FEC on. Measured within 0.4 dB above it; 60 ms CBR fell 1.6 dB below while
/// a large LBRR section still came out of the first frame's share alone.
const NORMAL_LEVEL_MARGIN_DB: f64 = 1.0;

/// In-band FEC must not desync the normal decode, must cost the main frame
/// no more than it costs libopus's encoder, and must let libopus recover each
/// lost packet from the next packet's LBRR section about as well as
/// libopus's own LBRR, at the right level.
fn check_fec_decodes_in_libopus(ch: usize, ms: usize, cbr: bool) {
    let input = voiced_16k(ch, ms);
    let rs_plain = encode_stream(Enc::Rust, &input, ch, ms, false, cbr);
    let rs = encode_stream(Enc::Rust, &input, ch, ms, true, cbr);
    let c_plain = encode_stream(Enc::Libopus, &input, ch, ms, false, cbr);
    let c = encode_stream(Enc::Libopus, &input, ch, ms, true, cbr);

    let rs_tax = mean(&rs_plain.normal) - mean(&rs.normal);
    let c_tax = mean(&c_plain.normal) - mean(&c.normal);
    let label = format!("{ms} ms {ch}ch {}", if cbr { "CBR" } else { "VBR" });
    println!(
        "{label} + FEC -> libopus: normal mean {:.1} dB, worst {:.1} dB, FEC tax {rs_tax:.1} dB \
         (libopus enc {c_tax:.1} dB); LBRR mean {:.1} dB, worst {:.1} dB (libopus enc mean {:.1} dB, worst {:.1} dB); \
         level against the input: normal {:+.1} dB, LBRR {:+.1} dB (libopus enc {:+.1} / {:+.1} dB); \
         LBRR in {}/{} packets (libopus enc {}/{}); \
         opus-rs vs libopus {:.1} dB",
        mean(&rs.normal),
        min(&rs.normal),
        mean(&rs.lbrr),
        min(&rs.lbrr),
        mean(&c.lbrr),
        min(&c.lbrr),
        median(&rs.normal_level),
        median(&rs.lbrr_level),
        median(&c.normal_level),
        median(&c.lbrr_level),
        rs.lbrr_packets,
        rs.packets,
        c.lbrr_packets,
        c.packets,
        min(&rs.agreement)
    );

    assert!(
        min(&rs.normal) > INPUT_FLOOR_DB,
        "{label} + FEC: normal decode desynced in libopus; per-window SNR: {:.1?}",
        rs.normal
    );
    assert!(
        rs_tax <= c_tax + FEC_MARGIN_DB,
        "{label}: FEC costs the normal decode {rs_tax:.1} dB, libopus's encoder {c_tax:.1} dB"
    );
    // A starved main frame decodes quiet before its SNR shows much.
    assert!(
        median(&rs.normal_level) >= median(&c.normal_level) - NORMAL_LEVEL_MARGIN_DB,
        "{label} + FEC: normal decode at {:+.1} dB against the input, libopus's encoder {:+.1} dB",
        median(&rs.normal_level),
        median(&c.normal_level)
    );
    // libopus's speech gate also skips frames, so compare packet counts with
    // its encoder rather than expecting LBRR in every packet.
    assert!(
        rs.lbrr_packets + LBRR_PACKETS_SLACK >= c.lbrr_packets,
        "{label}: LBRR in {}/{} packets, libopus's encoder {}/{}",
        rs.lbrr_packets,
        rs.packets,
        c.lbrr_packets,
        c.packets
    );
    assert!(
        min(&rs.lbrr) > LBRR_FLOOR_DB && mean(&rs.lbrr) >= mean(&c.lbrr) - FEC_MARGIN_DB,
        "{label}: LBRR recovery {:.1} dB mean against libopus's {:.1} dB; per-window SNR: {:.1?}",
        mean(&rs.lbrr),
        mean(&c.lbrr),
        rs.lbrr
    );
    let level = median(&rs.lbrr_level);
    assert!(
        (LBRR_LEVEL_DB.0..=LBRR_LEVEL_DB.1).contains(&level),
        "{label}: LBRR plays {level:+.1} dB against the input"
    );
    assert!(
        min(&rs.agreement) > MATCH_DB,
        "decoders disagree: {:.1?}",
        rs.agreement
    );
}

#[test]
fn opus_rs_mono_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, 20, false);
}

#[test]
fn opus_rs_stereo_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(2, 20, false);
}

/// Issue #36: at CBR the LBRR copy of the main frame never fit the packet, so
/// no LBRR was ever written and libopus could only conceal a lost packet.
#[test]
fn opus_rs_mono_cbr_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, 20, true);
}

#[test]
fn opus_rs_stereo_cbr_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(2, 20, true);
}

#[test]
fn opus_rs_40ms_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, 40, true);
    check_fec_decodes_in_libopus(1, 40, false);
}

#[test]
fn opus_rs_60ms_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, 60, true);
    check_fec_decodes_in_libopus(1, 60, false);
}

/// Level against the input (dB) of the recovered SILK frame 2 of every 60 ms
/// CBR packet after the first, for opus-rs's and libopus's encoders, when
/// each packet starts with `silent_ms` of silence and the rest is voice.
/// Only well-voiced windows count: near the minima of the test signal's 3 Hz
/// envelope the speech gate leaves frames without LBRR in both encoders, and
/// libopus conceals them instead.
fn frame_2_lbrr_levels(silent_ms: usize) -> (Vec<f64>, Vec<f64>) {
    let (ms, win) = (60, 320);
    let frame = 16 * ms;
    let mut input = voiced_16k(1, ms);
    for p in 0..input.len() / frame {
        input[p * frame..p * frame + 16 * silent_ms].fill(0.0);
    }
    let rs = encode_stream(Enc::Rust, &input, 1, ms, true, true);
    let c = encode_stream(Enc::Libopus, &input, 1, ms, true, true);
    let input_level = window_levels(&vec![1.0; input.len()], &input, win, frame);
    let loudest = input_level
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let pick = |levels: &[f64]| -> Vec<f64> {
        (2..levels.len())
            .step_by(3)
            .filter(|&w| input_level[w] > loudest - 6.0)
            .map(|w| levels[w])
            .collect()
    };
    let levels = (pick(&rs.lbrr_level), pick(&c.lbrr_level));
    println!(
        "60 ms packets, {silent_ms} ms silence then voice + FEC, LBRR level of SILK frame 2 \
         against the input: opus-rs {:+.1?} dB; libopus enc {:+.1?} dB",
        levels.0, levels.1
    );
    assert!(
        levels.0.len() >= 5,
        "too few voiced windows: {:?}",
        levels.0
    );
    levels
}

/// Recovered frames sit at the input's level: typically within
/// `LBRR_EDGE_MEDIAN_DB`, and none beyond `LBRR_EDGE_MAX_DB`.
fn assert_at_input_level(levels: &[f64], what: &str) {
    let worst = levels.iter().fold(0f64, |w, l| w.max(l.abs()));
    assert!(
        median(levels).abs() <= LBRR_EDGE_MEDIAN_DB && worst <= LBRR_EDGE_MAX_DB,
        "{what} at the wrong level: {levels:+.1?} dB"
    );
}

/// An LBRR frame whose predecessor in the packet carries no LBRR is coded
/// independently, so its first gain index is read as an absolute level. The
/// main frame's index there is a delta from the previous frame; libopus
/// `silk_LBRR_encode` raises that delta and writes it as an absolute level.
/// 40 ms of silence leaves SILK frame 1 below the LBRR speech gate, so frame
/// 2 is in exactly that position in every packet.
#[test]
fn opus_rs_lbrr_after_a_frame_without_lbrr_keeps_its_level() {
    let (levels, _) = frame_2_lbrr_levels(40);
    assert_at_input_level(&levels, "independently coded LBRR frames");
}

/// An LBRR frame that follows another LBRR frame is coded as a delta from it.
/// libopus reuses the main frame's delta, which is taken from the previous
/// *main* frame; when the rate loop re-quantized that frame after its LBRR copy
/// was made, the two histories part and the LBRR frame lands many dB off. A
/// 60 ms CBR packet whose LBRR section fills the first frame's share forces
/// that every packet: 20 ms of silence, then 40 ms of voice whose LBRR comes
/// at the start of the next packet.
#[test]
fn opus_rs_lbrr_after_a_requantized_main_frame_keeps_its_level() {
    let (levels, _) = frame_2_lbrr_levels(20);
    assert_at_input_level(&levels, "conditionally coded LBRR frames");
}

/// How far recovered frames may sit from the input in those two cases.
/// Measured, opus-rs: medians -0.6 and -0.1 dB, every window within 1.7 dB.
/// libopus's encoder: median -5.3 dB after a frame without LBRR, and windows
/// up to +19 dB after a re-quantized main frame.
const LBRR_EDGE_MEDIAN_DB: f64 = 1.5;
const LBRR_EDGE_MAX_DB: f64 = 3.0;
