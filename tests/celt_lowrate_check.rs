use opus_rs::{Application, OpusDecoder, OpusEncoder};

#[test]
fn celt_at_8_and_16k() {
    for (sr, bps) in [(8000usize, 5000i32), (16000, 5000), (16000, 3000), (8000, 3000)] {
        let frame = sr * 20 / 1000;
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Audio).unwrap();
        enc.bitrate_bps = bps;
        enc.use_cbr = true;
        let mut dec = OpusDecoder::new(sr as i32, 1).unwrap();
        let input: Vec<f32> = (0..frame * 20)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (0.3 * (2.0 * std::f32::consts::PI * 300.0 * t).sin())
            })
            .collect();
        let mut ok = 0;
        for fr in 0..20 {
            let mut pkt = vec![0u8; 1500];
            match enc.encode(&input[fr * frame..(fr + 1) * frame], frame, &mut pkt) {
                Ok(n) => {
                    let mut out = vec![0f32; frame];
                    if dec.decode(&pkt[..n], frame, &mut out).is_ok() {
                        ok += 1;
                    }
                }
                Err(e) => {
                    println!("{sr} {bps}: encode ERR {e}");
                    break;
                }
            }
        }
        println!("{sr} Hz {bps} bps: {ok}/20 frames encoded+decoded, last toc {:#04x}", {
            let mut pkt = vec![0u8; 1500];
            let _ = enc.encode(&input[..frame], frame, &mut pkt).map(|n| pkt[0]);
            pkt[0]
        });
    }
}
