use crate::range_coder::RangeCoder;
use crate::silk::control_fixed::*;
use crate::silk::control_snr::silk_control_snr;
use crate::silk::define::*;
use crate::silk::encode_indices::*;
use crate::silk::encode_pulses::*;
use crate::silk::gain_quant::{
    silk_gains_dequant, silk_gains_encode_levels, silk_gains_id, silk_gains_levels,
    silk_gains_quant,
};
use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use crate::silk::lp_variable_cutoff::*;
use crate::silk::macros::*;
use crate::silk::noise_shape_analysis::*;
use crate::silk::nsq::*;
use crate::silk::nsq_del_dec::*;
use crate::silk::pitch_analysis::*;
use crate::silk::structs::*;
use crate::silk::vad::silk_vad_get_sa_q8;

pub fn silk_encode_do_vad(ps_enc: &mut SilkEncoderState, input: &[i16], activity: i32) {
    let activity_threshold = SPEECH_ACTIVITY_DTX_THRES_Q8;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    silk_vad_get_sa_q8(ps_enc, input, frame_length);

    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }

    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX
        {
            ps_enc.s_cmn.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            ps_enc.s_cmn.in_dtx = 0;
        }
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 0;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 1;
    }
}

pub fn silk_encode_prefill(ps_enc: &mut SilkEncoderState, samples: &[i16], _activity: i32) {
    let fs_khz = ps_enc.s_cmn.fs_khz as usize;

    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return;
    }

    let prefill_frame_length = fs_khz * 10;

    if samples.len() < prefill_frame_length {
        return;
    }

    let real_frame_length = ps_enc.s_cmn.frame_length as usize;
    let real_nb_subfr = ps_enc.s_cmn.nb_subfr;
    let real_subfr_length = ps_enc.s_cmn.subfr_length;

    ps_enc.s_cmn.frame_length = prefill_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = 2;
    ps_enc.s_cmn.subfr_length = (prefill_frame_length / 2) as i32;

    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape_ms_samples = 5 * fs_khz;

    let n = prefill_frame_length.min(samples.len());

    let mut input_buf = [0i16; super::define::MAX_FRAME_LENGTH + 2];
    input_buf[0] = ps_enc.stereo.s_mid[0];
    input_buf[1] = ps_enc.stereo.s_mid[1];
    input_buf[2..2 + n].copy_from_slice(&samples[..n]);
    ps_enc.stereo.s_mid[0] = input_buf[prefill_frame_length];
    ps_enc.stereo.s_mid[1] = input_buf[prefill_frame_length + 1];

    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut input_buf[1..],
        prefill_frame_length,
    );

    let x_frame_idx = ltp_mem_length;
    let dst = x_frame_idx + la_shape_ms_samples;

    if dst + prefill_frame_length <= ps_enc.s_cmn.x_buf.len() {
        ps_enc.s_cmn.x_buf[dst..dst + prefill_frame_length]
            .copy_from_slice(&input_buf[1..1 + prefill_frame_length]);
    }

    let move_len = ltp_mem_length + la_shape_ms_samples;
    if prefill_frame_length + move_len <= ps_enc.s_cmn.x_buf.len() {
        ps_enc
            .s_cmn
            .x_buf
            .copy_within(prefill_frame_length..prefill_frame_length + move_len, 0);
    }

    ps_enc.s_cmn.frame_length = real_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = real_nb_subfr;
    ps_enc.s_cmn.subfr_length = real_subfr_length;
}

/// Noise-shaping quantization of one frame with the quantizer the complexity
/// setting selects: delayed decision when it keeps more than one state, plain
/// NSQ otherwise. Returns the seed to code with the pulses (delayed decision
/// picks the winning state's).
#[allow(clippy::too_many_arguments)]
fn silk_nsq_dispatch(
    cmn: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    indices: &SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    ctrl: &SilkEncoderControl,
    gains_q16: &[i32],
) -> i8 {
    let mut pred_coef_q12 = [0i16; 2 * MAX_LPC_ORDER];
    pred_coef_q12[..MAX_LPC_ORDER].copy_from_slice(&ctrl.pred_coef_q12[0]);
    pred_coef_q12[MAX_LPC_ORDER..].copy_from_slice(&ctrl.pred_coef_q12[1]);

    if cmn.n_states_delayed_decision > 1 {
        silk_nsq_del_dec(
            cmn,
            nsq,
            indices,
            x16,
            pulses,
            &pred_coef_q12,
            &ctrl.ltp_coef_q14,
            &ctrl.ar_q13,
            &ctrl.harm_shape_gain_q14,
            &ctrl.tilt_q14,
            &ctrl.lf_shp_q14,
            gains_q16,
            &ctrl.pitch_l,
            ctrl.lambda_q10,
            ctrl.ltp_scale_q14,
        ) as i8
    } else {
        silk_nsq(
            cmn,
            nsq,
            indices,
            x16,
            pulses,
            &pred_coef_q12,
            &ctrl.ltp_coef_q14,
            &ctrl.ar_q13,
            &ctrl.harm_shape_gain_q14,
            &ctrl.tilt_q14,
            &ctrl.lf_shp_q14,
            gains_q16,
            &ctrl.pitch_l,
            ctrl.lambda_q10,
            ctrl.ltp_scale_q14,
        );
        indices.seed
    }
}

/// Low-bitrate redundancy (libopus `silk_LBRR_encode_FIX`): quantize this
/// frame a second time, with the main frame's analysis but coarser gains, for
/// the next packet to carry. A decoder that lost this packet recovers the
/// frame from that copy.
///
/// The copy is a requantization, not a reuse of the main frame's pulses: it
/// runs the quantizer on a scratch copy of the NSQ state at the raised gains,
/// so it is cheaper than the main frame and plays at the input's level.
///
/// One intended difference from libopus, in how the raised gains are coded.
/// The decoder reads LBRR gain indices against the LBRR frames' own history:
/// a frame's first index is absolute when the previous frame of the packet
/// has no LBRR, and a delta from the previous LBRR frame otherwise. libopus
/// copies the main frame's indices, which are deltas from the previous *main*
/// frame, and raises the first one. That only lands on the intended level
/// while the two histories agree. When the main frame is coded conditionally
/// but its LBRR copy independently, a delta is read as an absolute level; and
/// when the rate loop re-quantizes a main frame after its LBRR copy was taken
/// (common at CBR), every later LBRR frame of the packet inherits the gap.
/// Either way the decoder recovers the frame many dB off (issue #36). Here the
/// main frame's levels, raised by the gain increase, are coded against the
/// LBRR history itself. Wherever libopus's copied indices do reach those
/// levels, they are the only indices that do, so the two agree.
#[inline(never)]
fn silk_lbrr_encode(
    ps_enc: &mut SilkEncoderState,
    ctrl: &SilkEncoderControl,
    x_frame_idx: usize,
    cond_coding: i32,
) {
    let n = ps_enc.s_cmn.n_frames_encoded as usize;
    if ps_enc.s_cmn.lbrr_enabled == 0
        || ps_enc.s_cmn.speech_activity_q8 <= LBRR_SPEECH_ACTIVITY_THRES_Q8
        || n >= MAX_FRAMES_PER_PACKET
    {
        return;
    }
    ps_enc.s_cmn.lbrr_flags[n] = 1;

    let mut indices = ps_enc.s_cmn.indices;
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let lbrr_cond_coding = if n > 0 && ps_enc.s_cmn.lbrr_flags[n - 1] != 0 {
        CODE_CONDITIONALLY
    } else {
        CODE_INDEPENDENTLY
    };
    if lbrr_cond_coding == CODE_INDEPENDENTLY {
        // The decoder clamps an absolute first index against its own last
        // level, which the encoder cannot see; like libopus, stand in this
        // frame's.
        ps_enc.s_cmn.lbrr_prev_last_gain_index = ps_enc.s_shape.last_gain_index;
    }

    // The main frame's gain levels as process_gains chose them, before the
    // rate loop, raised by the gain increase.
    let mut levels = silk_gains_levels(
        &indices.gains_indices,
        ctrl.last_gain_index_prev,
        (cond_coding == CODE_CONDITIONALLY) as i32,
        nb_subfr,
    );
    for level in &mut levels[..nb_subfr] {
        *level = (*level as i32 + ps_enc.s_cmn.lbrr_gain_increases).min(N_LEVELS_QGAIN - 1) as i8;
    }
    let lbrr_conditional = (lbrr_cond_coding == CODE_CONDITIONALLY) as i32;
    indices.gains_indices = silk_gains_encode_levels(
        &levels,
        ps_enc.s_cmn.lbrr_prev_last_gain_index,
        lbrr_conditional,
        nb_subfr,
    );

    // Gains as the decoder will reconstruct them from the LBRR indices.
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    silk_gains_dequant(
        &mut gains_q16,
        &indices.gains_indices,
        &mut ps_enc.s_cmn.lbrr_prev_last_gain_index,
        lbrr_conditional,
        nb_subfr,
    );

    let mut nsq = ps_enc.s_nsq;
    let mut pulses = [0i8; MAX_FRAME_LENGTH];
    indices.seed = silk_nsq_dispatch(
        &ps_enc.s_cmn,
        &mut nsq,
        &indices,
        &ps_enc.s_cmn.x_buf[x_frame_idx..],
        &mut pulses,
        ctrl,
        &gains_q16,
    );
    ps_enc.s_cmn.indices_lbrr[n] = indices;
    ps_enc.s_cmn.pulses_lbrr[n] = pulses;
}

pub fn silk_encode_frame(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    rc: &mut RangeCoder,
    pn_bytes_out: &mut i32,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut s_enc_ctrl = SilkEncoderControl::default();

    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape = ps_enc.s_cmn.la_shape as usize;

    let x_frame_idx = ltp_mem_length;

    let la_shape_max = 5 * ps_enc.s_cmn.fs_khz as usize;
    let new_samples_idx = x_frame_idx + la_shape_max;
    ps_enc.s_cmn.x_buf[new_samples_idx..new_samples_idx + frame_length]
        .copy_from_slice(&input[..frame_length]);

    let x_buf_copy = ps_enc.s_cmn.x_buf;

    let mut res_pitch = [0i16; LA_PITCH_MAX + MAX_FRAME_LENGTH + LTP_MEM_LENGTH_MS * MAX_FS_KHZ];
    let res_pitch_frame_idx = ltp_mem_length;

    silk_find_pitch_lags_fix(ps_enc, &mut s_enc_ctrl, &mut res_pitch, &x_buf_copy, 0);

    let x_tmp = &x_buf_copy[x_frame_idx - la_shape..];
    silk_noise_shape_analysis_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch[res_pitch_frame_idx..],
        x_tmp,
    );

    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let x_tmp_frame = &x_buf_copy[x_frame_idx - predict_lpc_order..];
    silk_find_pred_coefs_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch,
        res_pitch_frame_idx,
        x_tmp_frame,
        &x_buf_copy,
        cond_coding,
    );

    silk_process_gains_fix(ps_enc, &mut s_enc_ctrl, cond_coding);

    silk_lbrr_encode(ps_enc, &s_enc_ctrl, x_frame_idx, cond_coding);

    let max_iter = 6;
    let mut gain_mult_q8: i32 = 256;
    let mut found_lower = false;
    let mut found_upper = false;
    #[allow(unused_assignments)]
    let mut n_bits: i32 = 0;
    let mut n_bits_lower: i32 = 0;
    let mut n_bits_upper: i32 = 0;
    let mut gain_mult_lower: i32 = 0;
    let mut gain_mult_upper: i32 = 0;
    let mut gains_id: i32 =
        silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    let mut gains_id_lower: i32 = -1;
    let mut gains_id_upper: i32 = -1;

    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    let rc_copy = rc.clone();
    let nsq_copy = ps_enc.s_nsq;
    let seed_copy = ps_enc.s_cmn.indices.seed;
    let ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
    let ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
    let mut rc_copy2: Option<RangeCoder> = None;
    let mut nsq_copy2: Option<SilkNSQState> = None;
    let mut ec_buf_copy = [0u8; 1275];
    let mut last_gain_index_copy2: i8 = 0;

    let mut gain_lock = [false; MAX_NB_SUBFR];
    let mut best_gain_mult = [256i32; MAX_NB_SUBFR];
    let mut best_sum = [i32::MAX; MAX_NB_SUBFR];

    for iter in 0..=max_iter {
        if gains_id == gains_id_lower {
            n_bits = n_bits_lower;
        } else if gains_id == gains_id_upper {
            n_bits = n_bits_upper;
        } else {
            if iter > 0 {
                *rc = rc_copy.clone();
                ps_enc.s_nsq = nsq_copy;
                ps_enc.s_cmn.indices.seed = seed_copy;
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
            }

            let seed = silk_nsq_dispatch(
                &ps_enc.s_cmn,
                &mut ps_enc.s_nsq,
                &ps_enc.s_cmn.indices,
                &ps_enc.s_cmn.x_buf[x_frame_idx..],
                &mut ps_enc.pulses,
                &s_enc_ctrl,
                &s_enc_ctrl.gains_q16,
            );
            ps_enc.s_cmn.indices.seed = seed;

            if iter == max_iter && !found_lower {
                rc_copy2 = Some(rc.clone());
            }

            silk_encode_indices(
                ps_enc,
                rc,
                ps_enc.s_cmn.n_frames_encoded as usize,
                false,
                cond_coding,
            );

            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices.signal_type as i32,
                ps_enc.s_cmn.indices.quant_offset_type as i32,
                &ps_enc.pulses,
                ps_enc.s_cmn.frame_length as usize,
            );

            n_bits = rc.tell();

            if iter == max_iter && !found_lower && n_bits > max_bits {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                }

                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                    ps_enc.s_cmn.indices.gains_indices[i] = 4;
                }
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev;
                }
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;

                ps_enc.pulses.fill(0);

                silk_encode_indices(
                    ps_enc,
                    rc,
                    ps_enc.s_cmn.n_frames_encoded as usize,
                    false,
                    cond_coding,
                );
                silk_encode_pulses(
                    rc,
                    ps_enc.s_cmn.indices.signal_type as i32,
                    ps_enc.s_cmn.indices.quant_offset_type as i32,
                    &ps_enc.pulses,
                    ps_enc.s_cmn.frame_length as usize,
                );

                n_bits = rc.tell();
            }

            if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                break;
            }
        }

        if iter == max_iter {
            if found_lower && (gains_id == gains_id_lower || n_bits > max_bits) {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                    let offs = rc.offs as usize;
                    rc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                }
                if let Some(nsq_c2) = &nsq_copy2 {
                    ps_enc.s_nsq = *nsq_c2;
                }
                ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
            }
            break;
        }

        if n_bits > max_bits {
            if !found_lower && iter >= 2 {
                s_enc_ctrl.lambda_q10 =
                    silk_add_rshift32(s_enc_ctrl.lambda_q10, s_enc_ctrl.lambda_q10, 1);
                found_upper = false;
                gains_id_upper = -1;
            } else {
                found_upper = true;
                n_bits_upper = n_bits;
                gain_mult_upper = gain_mult_q8;
                gains_id_upper = gains_id;
            }
        } else if n_bits < max_bits - bits_margin {
            found_lower = true;
            n_bits_lower = n_bits;
            gain_mult_lower = gain_mult_q8;
            if gains_id != gains_id_lower {
                gains_id_lower = gains_id;

                rc_copy2 = Some(rc.clone());
                let offs = rc.offs as usize;
                ec_buf_copy[..offs].copy_from_slice(&rc.buf[..offs]);
                nsq_copy2 = Some(ps_enc.s_nsq);
                last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
            }
        } else {
            break;
        }

        if !found_lower && n_bits > max_bits {
            let subfr_length = ps_enc.s_cmn.subfr_length as usize;
            for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                let mut sum: i32 = 0;
                for j in (i * subfr_length)..((i + 1) * subfr_length) {
                    sum += ps_enc.pulses[j].abs() as i32;
                }
                if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                    best_sum[i] = sum;
                    best_gain_mult[i] = gain_mult_q8;
                } else {
                    gain_lock[i] = true;
                }
            }
        }

        if !(found_lower && found_upper) {
            if n_bits > max_bits {
                gain_mult_q8 = silk_min_32(1024, (gain_mult_q8 * 3) / 2);
            } else {
                gain_mult_q8 = silk_max_32(64, (gain_mult_q8 * 4) / 5);
            }
        } else {
            let delta = gain_mult_upper - gain_mult_lower;
            gain_mult_q8 = gain_mult_lower
                + silk_div32_16(
                    (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower),
                    n_bits_upper - n_bits_lower,
                );

            let lower_limit = silk_add_rshift32(gain_mult_lower, delta, 2);
            let upper_limit = silk_sub_rshift32(gain_mult_upper, delta, 2);
            if gain_mult_q8 > lower_limit {
                gain_mult_q8 = lower_limit;
            } else if gain_mult_q8 < upper_limit {
                gain_mult_q8 = upper_limit;
            }
        }

        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            let tmp = if gain_lock[i] {
                best_gain_mult[i]
            } else {
                gain_mult_q8
            };
            s_enc_ctrl.gains_q16[i] =
                silk_lshift_sat32(silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp), 8);
        }

        ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
        silk_gains_quant(
            &mut ps_enc.s_cmn.indices.gains_indices,
            &mut s_enc_ctrl.gains_q16,
            &mut ps_enc.s_shape.last_gain_index,
            if cond_coding == CODE_CONDITIONALLY {
                1
            } else {
                0
            },
            ps_enc.s_cmn.nb_subfr as usize,
        );

        gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    }

    let move_len = ltp_mem_length + 5 * ps_enc.s_cmn.fs_khz as usize;
    ps_enc
        .s_cmn
        .x_buf
        .copy_within(frame_length..frame_length + move_len, 0);

    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[ps_enc.s_cmn.nb_subfr as usize - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type as i32;
    ps_enc.s_cmn.first_frame_after_reset = 0;

    *pn_bytes_out = (rc.tell() + 7) >> 3;

    0
}

/// Encode the LBRR (in-band FEC) section for a whole packet. Matches libopus
/// enc_API.c: an LBRR symbol for multi-frame packets, then per frame a stereo
/// header (when stereo) followed by indices and pulses.
fn encode_lbrr_section(
    rc: &mut RangeCoder,
    ps_enc: &mut SilkEncoderState,
    lbrr_symbol: i32,
    n_frames_per_packet: i32,
) {
    if n_frames_per_packet > 1 {
        let lbrr_icdf = match n_frames_per_packet {
            2 => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
            3 => &crate::silk::tables::SILK_LBRR_FLAGS_3_ICDF[..],
            _ => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
        };
        rc.encode_icdf(lbrr_symbol - 1, lbrr_icdf, 8);
    }

    for i in 0..n_frames_per_packet as usize {
        if ps_enc.s_cmn.lbrr_flags[i] != 0 {
            // The speech-activity gate in silk_lbrr_encode only passes active
            // frames, and the LBRR index coder has no symbol for inactive ones.
            debug_assert!(ps_enc.s_cmn.indices_lbrr[i].signal_type >= TYPE_UNVOICED as i8);
            let lbrr_cond = if i > 0 && ps_enc.s_cmn.lbrr_flags[i - 1] != 0 {
                CODE_CONDITIONALLY
            } else {
                CODE_INDEPENDENTLY
            };
            // libopus writes the stereo header (pred + conditional mid-only
            // flag) before every LBRR payload; the decoder's LBRR skip path
            // expects it (issue #27, LBRR section).
            if ps_enc.s_cmn.n_channels == 2 {
                silk_encode_stereo(rc, 0, 0, 1);
            }
            silk_encode_indices(ps_enc, rc, i, true, lbrr_cond);
            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices_lbrr[i].signal_type as i32,
                ps_enc.s_cmn.indices_lbrr[i].quant_offset_type as i32,
                &ps_enc.s_cmn.pulses_lbrr[i],
                ps_enc.s_cmn.frame_length as usize,
            );
        }
    }
}

/// Bits to leave each frame after the LBRR section. When the rate loop cannot
/// fit a frame it keeps the frame's analysis indices and codes no pulses; that
/// measured at most 107 bits (voiced wideband) over 8-16 kHz, 10/20 ms frames
/// and noise, tone and mixed input. An LBRR section that would leave less is
/// dropped instead of letting a frame overflow the packet.
const MIN_FRAME_BITS: i32 = 128;

/// Per-frame stereo header (predictor and mid-only flag), on top of
/// `MIN_FRAME_BITS`.
const STEREO_HEADER_BITS: i32 = 16;

/// Start the packet with the LBRR copies of the previous packet's frames
/// (libopus enc_API.c), as long as they leave room for this packet's own
/// frames, then clear the flags for this packet's frames to set. Returns the
/// bits the section took.
fn silk_encode_lbrr_data(rc: &mut RangeCoder, ps_enc: &mut SilkEncoderState, max_bits: i32) -> i32 {
    let n_frames = ps_enc.s_cmn.n_frames_per_packet;
    let mut lbrr_symbol = 0;
    for i in 0..n_frames as usize {
        lbrr_symbol |= ps_enc.s_cmn.lbrr_flags[i] << i;
    }

    let start_bits = rc.tell();
    if lbrr_symbol > 0 {
        let saved_ec_prev_signal_type = ps_enc.s_cmn.ec_prev_signal_type;
        let saved_ec_prev_lag_index = ps_enc.s_cmn.ec_prev_lag_index;
        let mut trial = rc.clone();
        encode_lbrr_section(&mut trial, ps_enc, lbrr_symbol, n_frames);

        let frame_bits = MIN_FRAME_BITS
            + if ps_enc.s_cmn.n_channels == 2 {
                STEREO_HEADER_BITS
            } else {
                0
            };
        let capacity_bits = max_bits.min(rc.storage as i32 * 8 - 8);
        if trial.tell() + n_frames * frame_bits <= capacity_bits {
            *rc = trial;
        } else {
            ps_enc.s_cmn.ec_prev_signal_type = saved_ec_prev_signal_type;
            ps_enc.s_cmn.ec_prev_lag_index = saved_ec_prev_lag_index;
            lbrr_symbol = 0;
        }
    }

    ps_enc.s_cmn.lbrr_flag = (lbrr_symbol > 0) as i8;
    ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
    rc.tell() - start_bits
}

/// Bits frame `frame_idx` of a `tot_blocks`-frame packet may fill the packet
/// up to. Capping the earlier frames keeps a share of the packet for the
/// later ones; otherwise a 60 ms packet's first two frames can leave the last
/// one less than even its no-pulse fallback costs, and the packet overflows.
///
/// libopus (enc_API.c) takes the 3/5, 2/5 and 3/4 shares of the whole packet,
/// LBRR section included, so a large LBRR section comes out of the first
/// frame's share alone and can leave it no bits for pulses at all. Here the
/// shares are of what the LBRR section leaves; without LBRR they are
/// libopus's.
fn silk_frame_max_bits(max_bits: i32, lbrr_bits: i32, tot_blocks: i32, frame_idx: i32) -> i32 {
    let share = |num: i32, den: i32| lbrr_bits + (max_bits - lbrr_bits) * num / den;
    match (tot_blocks, frame_idx) {
        (2, 0) => share(3, 5),
        (3, 0) => share(2, 5),
        (3, 1) => share(3, 4),
        _ => max_bits,
    }
}

pub fn silk_encode(
    ps_enc: &mut SilkEncoderState,
    samples_in: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
) -> i32 {
    // The frame loop slices samples_in[..frame_end] based on this declared
    // length; a mismatched declaration would panic mid-encode
    // (issue #27 deep scan).
    assert!(
        n_samples_in <= samples_in.len(),
        "silk_encode: n_samples_in ({}) exceeds samples_in.len() ({})",
        n_samples_in,
        samples_in.len()
    );
    let n_frames_per_packet = ps_enc.s_cmn.n_frames_per_packet;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let packet_size_ms = ps_enc.s_cmn.packet_size_ms;

    ps_enc.s_cmn.n_frames_encoded = 0;

    // A reset encoder has no previous packet to protect (libopus enc_API.c).
    if ps_enc.s_cmn.first_frame_after_reset != 0 {
        ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
    }

    let n_blocks_of_10ms = (100 * n_samples_in as i32) / (ps_enc.s_cmn.fs_khz * 1000);
    let _tot_blocks = if n_blocks_of_10ms > 1 {
        n_blocks_of_10ms >> 1
    } else {
        1
    };

    let n_bits_total = target_rate_bps * packet_size_ms / 1000;

    // Room for the VAD and LBRR flags, patched in once the frames are coded,
    // then the LBRR section for the previous packet's frames.
    let n_channels = ps_enc.s_cmn.n_channels;
    let n_flag_bits = ((n_frames_per_packet + 1) * n_channels) as u32;
    let icdf = [(256i32 - (256i32 >> n_flag_bits)) as u8, 0u8];
    rc.encode_icdf(0, &icdf, 8);
    let lbrr_bits = silk_encode_lbrr_data(rc, ps_enc, max_bits);

    let mut sample_offset = 0usize;

    for frame_idx in 0..n_frames_per_packet {
        if frame_idx == 0 {
            silk_hp_variable_cutoff(&mut ps_enc.s_cmn);
        }

        let frame_end = (sample_offset + frame_length).min(n_samples_in);
        let raw_frame = &samples_in[sample_offset..frame_end];

        let fs_in_khz = ps_enc.s_cmn.fs_khz as usize;

        if raw_frame.len() < fs_in_khz {
            sample_offset += frame_length;
            continue;
        }

        let input_delay: usize = match fs_in_khz {
            8 => 6,
            12 => 7,
            16 => 10,
            24 => 6,
            48 => 12,
            _ => 0,
        };
        let n_samp: usize = fs_in_khz - input_delay;

        let n = raw_frame.len();

        if n > MAX_FRAME_LENGTH {
            sample_offset += frame_length;
            continue;
        }

        let mut resampler_out = [0i16; MAX_FRAME_LENGTH];

        let mut delay_buf = ps_enc.resampler_delay_buf;
        delay_buf[input_delay..fs_in_khz].copy_from_slice(&raw_frame[..n_samp]);

        resampler_out[..fs_in_khz].copy_from_slice(&delay_buf[..fs_in_khz]);

        let rest_len = n - fs_in_khz;
        let rest_end = n_samp + rest_len;

        if rest_end <= raw_frame.len() && n <= MAX_FRAME_LENGTH {
            resampler_out[fs_in_khz..n].copy_from_slice(&raw_frame[n_samp..rest_end]);
        }

        if n >= input_delay {
            delay_buf[..input_delay].copy_from_slice(&raw_frame[n - input_delay..]);
        }
        ps_enc.resampler_delay_buf = delay_buf;

        let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
        input_buf[0] = ps_enc.stereo.s_mid[0];
        input_buf[1] = ps_enc.stereo.s_mid[1];
        input_buf[2..2 + n].copy_from_slice(&resampler_out[..n]);

        ps_enc.stereo.s_mid[0] = input_buf[frame_length];
        ps_enc.stereo.s_mid[1] = input_buf[frame_length + 1];

        // Target rate (libopus enc_API.c, 1.5+): the LBRR section's size,
        // averaged over packets, comes out of the first frame's share of the
        // packet; later frames get their full share.
        let curr_lbrr_bits = if frame_idx == 0 { lbrr_bits } else { 0 };
        ps_enc.n_bits_used_lbrr = if curr_lbrr_bits < 10 {
            0
        } else if ps_enc.n_bits_used_lbrr < 10 {
            curr_lbrr_bits
        } else {
            (ps_enc.n_bits_used_lbrr + curr_lbrr_bits) / 2
        };
        let n_bits = (n_bits_total - ps_enc.n_bits_used_lbrr) / n_frames_per_packet;
        let frame_rate_bps = n_bits * if packet_size_ms == 10 { 100 } else { 50 };
        // Never exceed the input bitrate (libopus silk_LIMIT, which takes its
        // bounds in either order).
        let frame_rate_bps =
            frame_rate_bps.clamp(target_rate_bps.min(5000), target_rate_bps.max(5000));
        silk_control_snr(&mut ps_enc.s_cmn, frame_rate_bps);

        let vad_frame = &input_buf[1..1 + frame_length];
        silk_encode_do_vad(ps_enc, vad_frame, activity);

        // Per-frame stereo header: mid/side prediction followed by the
        // mid-only flag (written when the side channel carries no VAD, which
        // is always the case for this port's mid-only stereo). The decoder
        // reads these bits for every frame (dec_API.c / dec_api.rs).
        if ps_enc.s_cmn.n_channels == 2 {
            silk_encode_stereo(rc, 0, 0, 1);
        }

        silk_lp_variable_cutoff(&mut ps_enc.s_cmn.s_lp, &mut input_buf[1..], frame_length);

        let frame_samples = &input_buf[1..1 + frame_length];

        let cond_coding = if ps_enc.s_cmn.n_frames_encoded == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        // silk_encode_frame compares this against the range coder's tell,
        // which already counts the flags and the LBRR section.
        let frame_max_bits = silk_frame_max_bits(max_bits, lbrr_bits, _tot_blocks, frame_idx);

        let mut frame_bytes = 0i32;
        let ret = silk_encode_frame(
            ps_enc,
            frame_samples,
            rc,
            &mut frame_bytes,
            cond_coding,
            frame_max_bits,
            if use_cbr != 0 && frame_idx == n_frames_per_packet - 1 {
                1
            } else {
                0
            },
        );
        if ret != 0 {
            return ret;
        }

        ps_enc.s_cmn.n_frames_encoded += 1;
        sample_offset += frame_length;
    }

    let mut flags = 0u32;
    for i in 0..n_frames_per_packet as usize {
        flags <<= 1;
        flags |= ps_enc.s_cmn.vad_flags[i] as u32;
    }
    flags <<= 1;
    flags |= ps_enc.s_cmn.lbrr_flag as u32;
    if n_channels == 2 {
        flags <<= (n_frames_per_packet + 1) as u32;
    }

    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;

    0
}
