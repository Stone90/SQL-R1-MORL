# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import sys
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def pc_grad_combine(grads_a, grads_b):
    """
    PC-Grad projection that computes the combined gradient in-place on
    grads_a, avoiding the allocation of separate projected-gradient lists.

    When gradients conflict (dot < 0):
        proj_a = g_a - (d / ||b||^2) g_b
        proj_b = g_b - (d / ||a||^2) g_a
        combined = proj_a + proj_b = (1 - d/||a||^2) g_a + (1 - d/||b||^2) g_b

    Returns (grads_a, conflict_flag) where grads_a[i] has been overwritten
    with the combined projected gradient.
    """
    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for g_a, g_b in zip(grads_a, grads_b):
        dot_product += torch.sum(g_a * g_b)
        norm_a_sq += torch.sum(g_a * g_a)
        norm_b_sq += torch.sum(g_b * g_b)

    if dot_product < 0:
        coeff_a = (1.0 - dot_product / (norm_a_sq + 1e-8)).item()
        coeff_b = (1.0 - dot_product / (norm_b_sq + 1e-8)).item()
        for g_a, g_b in zip(grads_a, grads_b):
            g_a.mul_(coeff_a).add_(g_b, alpha=coeff_b)
        return grads_a, 1.0

    for g_a, g_b in zip(grads_a, grads_b):
        g_a.add_(g_b)
    return grads_a, 0.0


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        # Deferred optimizer offload: load states to GPU just before step
        is_offload = self.config.fsdp_config.get('optimizer_offload', False)
        if is_offload:
            from verl.utils.fsdp_utils import load_fsdp_optimizer
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()

        # Immediately offload optimizer states back to CPU
        if is_offload:
            from verl.utils.fsdp_utils import offload_fsdp_optimizer
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        if not self.config.get('use_dynamic_bsz', False):
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = self.config.get('use_dynamic_bsz', False)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if "accuracy_rewards" in data.batch:
            select_keys.extend(['accuracy_rewards', 'efficiency_rewards'])
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        batch = data.select(batch_keys=select_keys).batch
        dataloader = list(batch.split(self.config.ppo_mini_batch_size))
        metrics = {}

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        def _diag(msg):
            rss_gb = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) / 1024 / 1024
            gpu_alloc = torch.cuda.memory_allocated() / 1024**3
            gpu_resv = torch.cuda.memory_reserved() / 1024**3
            print(f"[rank{rank}][diag] {msg} | RSS={rss_gb:.1f}GB GPU_alloc={gpu_alloc:.1f}GB GPU_resv={gpu_resv:.1f}GB", file=sys.stderr, flush=True)
        _diag(f"update_policy start, {len(dataloader)} mini-batches")

        for batch_idx, mini_batch in enumerate(dataloader):
            if use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)
            self.gradient_accumulation = len(micro_batches)
            _diag(f"mini-batch {batch_idx}, {len(micro_batches)} micro-batches")
            self.actor_optimizer.zero_grad()

            for data_micro in micro_batches:
                data_micro = data_micro.cuda()
                responses = data_micro['responses']
                response_length = responses.size(1)
                attention_mask = data_micro['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data_micro['old_log_probs']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                if self.config.get("enable_pc_grad", False) and "accuracy_rewards" in data_micro:
                    acc_adv = data_micro['accuracy_rewards']
                    eff_adv = data_micro['efficiency_rewards']

                    # Fast path: skip dual-pass PC-Grad when efficiency signal is trivially zero
                    if torch.all(eff_adv == 0):
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data_micro, temperature=temperature)
                        entropy_loss = verl_F.masked_mean(entropy, response_mask)

                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob, log_prob, acc_adv, response_mask, clip_ratio)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = data_micro['ref_log_prob']
                            kld = core_algos.kl_penalty(logprob=log_prob,
                                                        ref_logprob=ref_log_prob,
                                                        kl_penalty=self.config.kl_loss_type)
                            kl_loss = masked_mean(kld, response_mask)
                            policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                            append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach().item()})
                            append_to_dict(metrics, {'actor/kl_coef': self.config.kl_loss_coef})

                        loss = policy_loss / self.gradient_accumulation
                        loss.backward()
                        append_to_dict(metrics, {'actor/pg_loss': pg_loss.detach().item(),
                                                 'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                                                 'actor/ppo_kl': ppo_kl.detach().item(),
                                                 'actor/pc_grad_fast_path': 1.0})

                    else:
                        # Full dual-pass PC-Grad path

                        # Save accumulated grads from prior micro-batches
                        first_grad = next(iter(self.actor_module.parameters())).grad
                        if first_grad is not None:
                            accumulated_grads = [p.grad.clone() for p in self.actor_module.parameters()]
                        else:
                            accumulated_grads = None
                        self.actor_optimizer.zero_grad()

                        # Pass 1: Accuracy (forward + backward, includes KL regulariser)
                        entropy_acc, log_prob_acc = self._forward_micro_batch(micro_batch=data_micro, temperature=temperature)
                        entropy_loss = verl_F.masked_mean(entropy_acc, response_mask)

                        kl_loss_term = torch.tensor(0.0, device=log_prob_acc.device)
                        if self.config.use_kl_loss:
                            ref_log_prob = data_micro['ref_log_prob']
                            kld = core_algos.kl_penalty(logprob=log_prob_acc,
                                                        ref_logprob=ref_log_prob,
                                                        kl_penalty=self.config.kl_loss_type)
                            kl_loss = masked_mean(kld, response_mask)
                            kl_loss_term = kl_loss * self.config.kl_loss_coef
                            append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach().item()})
                            append_to_dict(metrics, {'actor/kl_coef': self.config.kl_loss_coef})

                        pg_loss_acc, pg_clipfrac_acc, _ = core_algos.compute_policy_loss(old_log_prob, log_prob_acc, acc_adv, response_mask, clip_ratio)
                        loss_acc = (pg_loss_acc - entropy_loss * entropy_coeff - kl_loss_term) / self.gradient_accumulation
                        _diag(f"  mb{batch_idx} pass1 backward")
                        loss_acc.backward()

                        # Clone pass-1 grads (keep on GPU — H100 has plenty of VRAM)
                        grad_acc = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.actor_module.parameters()]
                        del entropy_acc, log_prob_acc, loss_acc

                        self.actor_optimizer.zero_grad()
                        torch.cuda.empty_cache()

                        # Pass 2: Efficiency (forward + backward, no KL -- applied once in accuracy pass)
                        entropy_eff, log_prob_eff = self._forward_micro_batch(micro_batch=data_micro, temperature=temperature)
                        entropy_loss_eff = verl_F.masked_mean(entropy_eff, response_mask)
                        pg_loss_eff, pg_clipfrac_eff, _ = core_algos.compute_policy_loss(old_log_prob, log_prob_eff, eff_adv, response_mask, clip_ratio)
                        loss_eff = (pg_loss_eff - entropy_loss_eff * entropy_coeff) / self.gradient_accumulation
                        _diag(f"  mb{batch_idx} pass2 backward")
                        loss_eff.backward()
                        del entropy_eff, log_prob_eff, loss_eff
                        torch.cuda.empty_cache()

                        _diag(f"  mb{batch_idx} pc-grad project")
                        # --- PC-Grad projection ---
                        # Phase 1: Compute dot product and norms
                        dot_product, norm_a_sq, norm_b_sq = 0.0, 0.0, 0.0
                        for i, p in enumerate(self.actor_module.parameters()):
                            g_a = grad_acc[i]
                            g_b = p.grad if p.grad is not None else torch.zeros_like(p)
                            dot_product += torch.sum(g_a * g_b).item()
                            norm_a_sq += torch.sum(g_a * g_a).item()
                            norm_b_sq += torch.sum(g_b * g_b).item()

                        if dot_product < 0:
                            coeff_a = 1.0 - dot_product / (norm_a_sq + 1e-8)
                            coeff_b = 1.0 - dot_product / (norm_b_sq + 1e-8)
                            conflict_val = 1.0
                        else:
                            coeff_a = 1.0
                            coeff_b = 1.0
                            conflict_val = 0.0

                        # Phase 2: Apply projection and restore accumulated grads
                        for i, p in enumerate(self.actor_module.parameters()):
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            p.grad.mul_(coeff_b).add_(grad_acc[i], alpha=coeff_a)
                            if accumulated_grads is not None:
                                p.grad.add_(accumulated_grads[i])
                        del grad_acc, accumulated_grads

                        append_to_dict(metrics, {'actor/pc_grad_conflict_rate': conflict_val})
                        append_to_dict(metrics, {'actor/pg_loss_acc': pg_loss_acc.detach().item()})
                        append_to_dict(metrics, {'actor/pg_loss_eff': pg_loss_eff.detach().item()})
                        append_to_dict(metrics, {'actor/pg_clipfrac_acc': pg_clipfrac_acc.detach().item()})
                        append_to_dict(metrics, {'actor/pg_clipfrac_eff': pg_clipfrac_eff.detach().item()})
                        append_to_dict(metrics, {'actor/entropy_loss_acc': entropy_loss.detach().item()})
                        append_to_dict(metrics, {'actor/entropy_loss_eff': entropy_loss_eff.detach().item()})

                else:
                    # Baseline single-objective path
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data_micro, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    advantages = data_micro['advantages']
                    pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, clip_ratio)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data_micro['ref_log_prob']
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)
                        policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                        append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach().item()})
                        append_to_dict(metrics, {'actor/kl_coef': self.config.kl_loss_coef})

                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    append_to_dict(metrics, {'actor/pg_loss': pg_loss.detach().item(),
                                             'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                                             'actor/ppo_kl': ppo_kl.detach().item()})

                append_to_dict(metrics, {'actor/entropy_loss': entropy_loss.detach().item()})

            torch.cuda.empty_cache()
            _diag(f"mini-batch {batch_idx} optimizer step")
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
