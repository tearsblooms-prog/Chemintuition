
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import math

CONDITION_NUM_STEPS = 20
CONDITION_FEAT_PER_STEP_ORIGINAL = 32
RCS_VOCAB_SIZE = 1035 + 1
OTHER_COND_VOCAB_SIZE = 14 + 1

FINGERPRINT_STEP_STRUCTURE = {
    "is_active": [0],
    "reagents": list(range(1, 6)),
    "catalysts": list(range(6, 10)),
    "solvents": list(range(10, 14)),
    "temp_time_segments": list(range(14, 30)),
    "other_conditions": list(range(30, 32)),
}


# --- Tokenizer ---
class ReactionConditionTokenizer:
    def __init__(self):
        self.vocab = self._build_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_id = self.vocab['[PAD]']
        self.bos_id = self.vocab['[BOS]']
        self.eos_id = self.vocab['[EOS]']
        self.sep_id = self.vocab['[SEP]']

    def _build_vocab(self):
        vocab = OrderedDict()

        def add_token(token):
            if token not in vocab: vocab[token] = len(vocab)

        for token in ['[PAD]', '[BOS]', '[EOS]', '[SEP]']: add_token(token)
        for i in range(RCS_VOCAB_SIZE): add_token(f'RCS_{i}')
        for i in range(OTHER_COND_VOCAB_SIZE): add_token(f'OC_{i}')
        self.temp_bins = []
        for t in range(-20, 31): self.temp_bins.append(f'TEMP_{t}')
        for t in range(-100, -20, 5): self.temp_bins.append(f'TEMP_{t}_{t + 5}')
        for t in range(31, 151, 5): self.temp_bins.append(f'TEMP_{t}_{t + 5}')
        for t in range(151, 301, 10): self.temp_bins.append(f'TEMP_{t}_{t + 10}')
        self.temp_bins += ['TEMP_rt', 'TEMP_undefined']
        for token in self.temp_bins: add_token(token)
        self.duration_bins = []
        for t in range(0, 120, 5): self.duration_bins.append(f'DUR_{t}-{t + 5}')
        for t in range(120, 480, 15): self.duration_bins.append(f'DUR_{t}-{t + 15}')
        for t in range(480, 2880, 60): self.duration_bins.append(f'DUR_{t}-{t + 60}')
        self.duration_bins.append('DUR_>2880')
        self.duration_bins.append('DUR_undefined')
        for token in self.duration_bins: add_token(token)
        for token in ['REFLUX_true', 'REFLUX_false']: add_token(token)
        return vocab

    def _discretize_temp(self, t):
        if t == -1:
            return 'TEMP_undefined'
        elif 18 <= t <= 28:
            return 'TEMP_rt'
        elif -20 <= t <= 30:
            return f'TEMP_{round(t)}'
        elif -100 <= t < -20:
            bin_start = math.floor(t / 5) * 5
            return f'TEMP_{bin_start}_{bin_start + 5}'
        elif 30 < t <= 150:
            bin_start = 31 + math.floor((t - 31) / 5) * 5
            return f'TEMP_{bin_start}_{bin_start + 5}'
        elif 150 < t <= 300:
            bin_start = 151 + math.floor((t - 151) / 10) * 10
            return f'TEMP_{bin_start}_{bin_start + 10}'
        elif t > 300:
            return 'TEMP_291_301'
        elif t < -100:
            return 'TEMP_-100_-95'
        else:
            return 'TEMP_undefined'

    def _discretize_duration(self, d):
        if d == -1:
            return 'DUR_undefined'
        elif d < 0:
            return 'DUR_0-5'
        elif d < 120:  # [0, 120)
            bin_start = math.floor(d / 5) * 5
            return f'DUR_{bin_start}-{bin_start + 5}'
        elif d < 480:  # [120, 480)
            bin_start = 120 + math.floor((d - 120) / 15) * 15
            return f'DUR_{bin_start}-{bin_start + 15}'
        elif d < 2880:  # [480, 2880)
            bin_start = 480 + math.floor((d - 480) / 60) * 60
            return f'DUR_{bin_start}-{bin_start + 60}'
        else:  # [2880, inf)
            return 'DUR_>2880'

    def _undiscretize_temp(self, token):
        if token == 'TEMP_undefined': return -1.0
        if token == 'TEMP_rt': return 25.0
        parts = token.split('_')
        try:
            if len(parts) == 2:
                return float(parts[1])
            elif len(parts) == 3:
                return (float(parts[1]) + float(parts[2])) / 2.0
        except (ValueError, IndexError):
            pass
        return -1.0

    def _undiscretize_duration(self, token):
        if token == 'DUR_undefined': return -1.0
        parts = token.split('_')
        try:
            if parts[1].startswith('>'): return float(parts[1][1:]) + 30
            range_parts = parts[1].split('-')
            if len(range_parts) == 2: return (float(range_parts[0]) + float(range_parts[1])) / 2.0
        except (ValueError, IndexError):
            pass
        return 0.0

    def fingerprint_to_tokens(self, fingerprint_640d):
        fingerprint_reshaped = fingerprint_640d.view(CONDITION_NUM_STEPS, CONDITION_FEAT_PER_STEP_ORIGINAL)
        tokens = [self.bos_id]
        for i in range(CONDITION_NUM_STEPS):
            if fingerprint_reshaped[i, 0].item() != 1.0: break
            step_tokens = []
            rcs_indices = FINGERPRINT_STEP_STRUCTURE['reagents'] + FINGERPRINT_STEP_STRUCTURE['catalysts'] + \
                          FINGERPRINT_STEP_STRUCTURE['solvents']
            for idx in rcs_indices:
                step_tokens.append(self.vocab[f'RCS_{int(fingerprint_reshaped[i, idx].item())}'])
            temp_time_raw = fingerprint_reshaped[i, FINGERPRINT_STEP_STRUCTURE['temp_time_segments']]
            for seg_idx in range(4):
                seg_data = temp_time_raw[seg_idx * 4: (seg_idx + 1) * 4]
                t_start, t_end, duration, is_reflux = seg_data[0].item(), seg_data[1].item(), seg_data[2].item(), \
                seg_data[3].item()

                step_tokens.append(self.vocab[self._discretize_temp(t_start)])
                step_tokens.append(self.vocab[self._discretize_temp(t_end)])

                step_tokens.append(self.vocab[self._discretize_duration(duration)])
                step_tokens.append(self.vocab['REFLUX_true' if is_reflux == 1.0 else 'REFLUX_false'])
            for idx in FINGERPRINT_STEP_STRUCTURE['other_conditions']:
                step_tokens.append(self.vocab[f'OC_{int(fingerprint_reshaped[i, idx].item())}'])
            tokens.extend(step_tokens)
            tokens.append(self.sep_id)
        tokens.append(self.eos_id)
        return torch.tensor(tokens, dtype=torch.long)

    def tokens_to_fingerprint(self, token_ids):
        token_ids = [tid.item() if torch.is_tensor(tid) else tid for tid in token_ids]
        token_ids = [tid for tid in token_ids if tid not in [self.bos_id, self.pad_id]]
        fingerprint = torch.zeros(CONDITION_NUM_STEPS, CONDITION_FEAT_PER_STEP_ORIGINAL)
        step_sequences = []
        current_step = []
        for tid in token_ids:
            if tid == self.sep_id:
                if current_step: step_sequences.append(current_step)
                current_step = []
            elif tid == self.eos_id:
                if current_step: step_sequences.append(current_step)
                break
            else:
                current_step.append(tid)
        for i, step_toks in enumerate(step_sequences):
            if i >= CONDITION_NUM_STEPS: break
            fingerprint[i, 0] = 1.0
            tok_idx = 0

            def get_next_token():
                nonlocal tok_idx
                if tok_idx < len(step_toks):
                    token = self.id_to_token[step_toks[tok_idx]]
                    tok_idx += 1
                    return token
                return None

            rcs_indices = FINGERPRINT_STEP_STRUCTURE['reagents'] + FINGERPRINT_STEP_STRUCTURE['catalysts'] + \
                          FINGERPRINT_STEP_STRUCTURE['solvents']
            for fp_idx in rcs_indices:
                token = get_next_token()
                if token and token.startswith('RCS_'):
                    fingerprint[i, fp_idx] = int(token.split('_')[1])
            temp_time_indices = FINGERPRINT_STEP_STRUCTURE['temp_time_segments']
            for seg_idx in range(4):
                fp_seg_start_idx = temp_time_indices[0] + seg_idx * 4

                t_start_tok, t_end_tok, dur_tok, reflux_tok = get_next_token(), get_next_token(), get_next_token(), get_next_token()

                if t_start_tok and t_end_tok and dur_tok and reflux_tok:
                    fingerprint[i, fp_seg_start_idx] = self._undiscretize_temp(t_start_tok)
                    fingerprint[i, fp_seg_start_idx + 1] = self._undiscretize_temp(t_end_tok)
                    fingerprint[i, fp_seg_start_idx + 2] = self._undiscretize_duration(dur_tok)
                    fingerprint[i, fp_seg_start_idx + 3] = 1.0 if reflux_tok == 'REFLUX_true' else 0.0
                else:
                    fingerprint[i, fp_seg_start_idx:fp_seg_start_idx + 4] = torch.tensor([-1.0, -1.0, -1.0, 0.0])

            oc_indices = FINGERPRINT_STEP_STRUCTURE['other_conditions']
            for fp_idx in oc_indices:
                token = get_next_token()
                if token and token.startswith('OC_'):
                    fingerprint[i, fp_idx] = int(token.split('_')[1])
        for i in range(len(step_sequences), CONDITION_NUM_STEPS):
            fingerprint[i, FINGERPRINT_STEP_STRUCTURE['temp_time_segments']] = torch.tensor(
                [-1.0, -1.0, -1.0, 0.0] * 4)
        return fingerprint.view(-1)