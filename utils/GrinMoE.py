import copy
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.moe.mappings import drop_tokens, gather_tokens
from deepspeed.moe.sharded_moe import FIRST_ALLTOALL_TIMER, MOE_TIMER, SECOND_ALLTOALL_TIMER, _AllToAll, einsum, gumbel_rsample
from deepspeed.utils import groups, log_dist
from deepspeed.utils.timer import SynchronizedWallClockTimer
from torch import Tensor
from transformers.activations import ACT2FN

from GrinQwenVL_v5_utils import compress_matrix, decompress_matrix


class SharedExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.shared_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class DynamicExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.dynamic_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class NULLExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_state):
        # return hidden_state * 0
        return torch.zeros_like(hidden_state, dtype=hidden_state.dtype, device=hidden_state.device)


#
# ---------------------------- copy from Grin ----------------------------
#


class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        scores: torch.Tensor,
        multiplier: torch.Tensor,
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
        ctx,
        grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expaned,
            None,
            None,
            None,
            None,
        )


def sparsemixer(scores, top_k, jitter_eps, training):
    masked_scores = scores
    multiplier_list = []
    selected_experts_list = []

    for _ in range(top_k):
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            # factor = scores.abs().clamp(min=mask_logits_threshold)
            factor = scores.abs().clamp(min=mask_logits_threshold.abs())  # Todo, 这里我主动加了一个.abs() 我认为factor这样才算比较好的尺度
            mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)  # jitter_noise

        # apply mask
        masked_gates = masked_scores.masked_fill(mask_logits_threshold, float("-inf"))

        if training:
            # 在训练期间使用 Gumbel 采样替代传统的 Top-k 选择，确保梯度的稳健性和随机性
            # Gumbel noise 使用 exponent() 和 log() 函数生成，用于从 masked_gates 中选择专家
            # V5 tip: 这里坑死了, 源代码的exponent和log生成会很大概率生成-inf, new_masked_gates全为-inf, 这里直接用了deepspeed的gumbel_rsample, Todo: 有时间再检查一下gumbel_rsample的原理
            noise = gumbel_rsample(masked_gates.shape, device=masked_gates.device)
            assert not torch.isnan(noise).any()
            new_masked_gates = masked_gates + noise
            selected_experts = (new_masked_gates).max(dim=-1)[1].unsqueeze(-1)  # gumbel sampling, more robust than than the multinomial method
        else:
            selected_experts = max_ind

        # compute scores for gradients
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

        if training:
            # compute midpoint mask
            max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
            mask_for_one = torch.logical_or(
                selected_experts == max_ind,
                torch.rand_like(max_scores) > 0.75,  # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
            )
            # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
            mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

            multiplier = mp.apply(
                scores,
                multiplier_o,
                selected_experts,
                masked_gates,
                mask_for_one,
            )
        else:
            multiplier = multiplier_o

        # masked out first expert
        masked_scores = torch.scatter(
            masked_scores,
            -1,
            selected_experts,
            float("-inf"),
        )

        multiplier_list.append(multiplier)
        selected_experts_list.append(selected_experts)

    multiplier = torch.concat(multiplier_list, dim=-1)
    selected_experts = torch.concat(selected_experts_list, dim=-1)

    # print(multiplier.shape)
    # if not ((multiplier != 0).sum(1) == top_k).all():
    #     iidx = ((multiplier != 0).sum(0) != top_k).nonzero().tolist()[:3]
    #     raise ValueError(f"top_k: {top_k} iidx: {iidx}\n multiplier: {[multiplier[xx[0]] for xx in iidx]}")

    # if torch.any(torch.sum(selected_experts.unsqueeze(-1) == selected_experts.unsqueeze(-2), dim=-1) > 1):
    #     iidx = (torch.sum(selected_experts.unsqueeze(-1) == selected_experts.unsqueeze(-2), dim=-1) > 1).nonzero().tolist()
    #     raise AssertionError(f"iidx: {iidx}, selected_experts: {[selected_experts[xx[0]] for xx in iidx]}, masked_scores: {[masked_scores[xx[0]] for xx in iidx]}")

    return (
        multiplier,
        selected_experts,
    )


def dynamic_expert_selection(logits, top_p):
    # logits (batch * sequence_length, dynamic_expert_num)
    dynamic_scores = torch.softmax(logits, dim=-1)
    dynamic_scores_sorted, _ = torch.sort(dynamic_scores, dim=-1, descending=True)
    dynamic_scores_cumsum = dynamic_scores_sorted.cumsum(dim=-1)
    dynamic_top_k = (~(dynamic_scores_cumsum >= top_p)).sum(dim=-1)
    dynamic_top_k = dynamic_top_k + 1
    return dynamic_top_k


def _capacity(num_tokens, num_experts, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


def cal_global_weight(
    expert_mask: torch.Tensor,
    full_router_logits: torch.Tensor,
    mlp_dynamic_expert_num: int,
    routing_weights: torch.Tensor,
):
    global_weight = torch.softmax(full_router_logits.masked_fill(expert_mask == 0, float("-inf")), dim=-1)
    # Todo: 这里的global weight和routing_weights 有多个融合方法
    # 1. 用scaling factor * routing_weights, 然后拼接上 fix_routing_weight (当前采用)
    # 2. 指定dynamic和fix的权重分配, 然后根据routing_weights和fix_routing_weight各自分配, 这个不太好
    global_dynamic_weight = global_weight[:, :mlp_dynamic_expert_num]
    global_fixed_weight = global_weight[:, mlp_dynamic_expert_num:]
    global_dynamic_weight = routing_weights * global_dynamic_weight.sum(-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1])  # 计算dynamic的weight缩放因数
    global_weight = torch.cat((global_dynamic_weight, global_fixed_weight), dim=-1)
    return global_weight


class GRINMoESparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.mlp_dynamic_expert_num = config.mlp_dynamic_expert_num + config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_real_expert_num = config.mlp_dynamic_expert_num
        self.mlp_dynamic_null_expert_num = config.mlp_dynamic_null_expert_num
        self.mlp_dynamic_top_p = config.mlp_dynamic_top_p
        self.mlp_dynamic_top_k = config.mlp_dynamic_top_k
        self.mlp_fixed_expert_num = config.mlp_fixed_expert_num
        self.num_experts = self.mlp_dynamic_expert_num + self.mlp_fixed_expert_num

        if self.mlp_dynamic_top_p == 0:
            print(f"mlp_dynamic_top_p is 0, will use mlp_dynamic_top_k={self.mlp_dynamic_top_k} instead !!!")

        self.ignore_differentiable_router = config.ignore_differentiable_router
        if self.ignore_differentiable_router:
            print("ignore_differentiable_router is True, will not use router_logits !!!")

        # gating & experts
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        # self.dynamic_null_experts = nn.ModuleList([NULLExpertMLP(config) for _ in range(self.mlp_dynamic_null_expert_num)]) # 没有用的专家
        self.fixed_real_moe = nn.ModuleList([SharedExpertMLP(config) for _ in range(self.mlp_fixed_expert_num)])
        # deepspeed moe for dynamic real expert
        self.dynamic_real_moe = MoE(config, DynamicExpertMLP(config), self.mlp_dynamic_real_expert_num, config.ep_size)

        # Jitter parameters
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise

        self.min_capacity = config.min_capacity
        self.capacity_factor = config.capacity_factor
        self.token_drop = config.token_drop
        self.drop_policy = config.drop_policy

        self.avg_hidden_states_last = config.avg_hidden_states_last
        self.drop_token_num_print = config.drop_token_num_print
        self.fp32_gate = config.fp32_gate

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, aux_balance_weight: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        original_hidden_states = hidden_states

        if self.training and self.fp32_gate:
            hidden_states = hidden_states.float()

        # input jitter_noise
        # Grin moe和deepspeed moe都会添加jitter_noise
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)

        hidden_states = hidden_states.view(-1, hidden_dim)

        # full_router_logits: (batch * sequence_length, n_experts)
        if self.training and self.fp32_gate:
            full_router_logits = torch.nn.functional.linear(hidden_states, weight=self.gate.weight.float(), bias=None)
        else:
            full_router_logits = self.gate(hidden_states)
        dynamic_router_logits = full_router_logits[:, : self.mlp_dynamic_expert_num]

        # 获得动态top_k
        if self.mlp_dynamic_top_p != 0:
            dynamic_top_k = dynamic_expert_selection(dynamic_router_logits, self.mlp_dynamic_top_p)
        else:
            dynamic_top_k = torch.full((dynamic_router_logits.shape[0],), self.mlp_dynamic_top_k, dtype=torch.int, device=dynamic_router_logits.device)

        # expert_mask:moe的路由情况 (batch * sequence_length, expert_num)
        expert_mask = torch.zeros((batch_size * sequence_length, self.num_experts), dtype=torch.int, device=hidden_states.device)

        #
        # ---------------- dynamic top_p experts ----------------
        #

        # 用来存grin路由的sparsemixer的返回值, 即group_routing_weights
        # Todo: group_routing_weights的sum和大概率和top k相同, 这样导致hidden state的权重和不等于1, 需要思考下
        routing_weights = torch.zeros((batch_size * sequence_length, self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
        for top_k in range(1, self.mlp_dynamic_expert_num + 1):
            # 获得当前top_k路由模式的token 位置
            group_idx = torch.nonzero(dynamic_top_k == top_k, as_tuple=True)[0]
            if len(group_idx) == 0:
                continue

            dynamic_group_logits = dynamic_router_logits[group_idx]
            # group_selected_experts: (group_batch_size, top_k), 按照优先级排序
            group_routing_weights, group_selected_experts = sparsemixer(
                dynamic_group_logits,
                top_k=top_k,
                jitter_eps=self.router_jitter_noise,
                training=self.training and not self.ignore_differentiable_router,
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            group_expert_mask = torch.nn.functional.one_hot(group_selected_experts, num_classes=self.num_experts)
            group_expert_mask = group_expert_mask.sum(dim=1)

            group_weight = torch.zeros((len(group_idx), self.mlp_dynamic_expert_num), dtype=hidden_states.dtype, device=hidden_states.device)
            group_weight.scatter_(dim=-1, index=group_selected_experts, src=group_routing_weights)
            routing_weights.index_add_(0, group_idx, group_weight)

            # 更新expert_mask
            # 设定0~self.mlp_dynamic_expert_num 为动态专家
            expert_mask.index_add_(0, group_idx, group_expert_mask.to(expert_mask.dtype))

        # 目前决定还是将routing_weights先求和归一化 # Todo: 看看效果
        routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        #
        # ---------------- attention mask ----------------
        #

        if attention_mask is not None:
            # [fix] padding token 的路由会影响aux balance loss的计算, 并且还会占capacity, 这里强制设置为0
            # 看了一下, aux balance loss 在设置aux balance loss的时候融合了attention mask (在GrinQwen2VLForConditionalGeneration的forward中), 这时候不会影响
            # 但是会影响capacity, 这里load_balancing_loss_func 传入capacity_expert_mask可能会导致padding经过softmax后为nan
            assert len(attention_mask.shape) == 2, f"{attention_mask.shape}"  # B, L
            attention_mask = attention_mask.to(expert_mask.dtype).view(-1).unsqueeze(-1).expand(-1, self.num_experts)
            expert_mask = expert_mask * attention_mask

        #
        # ---------------- fixed top_p experts ----------------
        #

        if self.mlp_dynamic_expert_num < self.num_experts:
            expert_mask[:, self.mlp_dynamic_expert_num :] = 1  # 只需要把expert mask 设为1即可

        #
        # ---------------- aux balance loss ----------------
        #

        # 这里选择在token drop前计算aux balance loss, 更加精确, 但是要重新算一下global weight
        aux_loss = load_balancing_loss_func(
            expert_mask=expert_mask,
            mlp_dynamic_expert_num=self.mlp_dynamic_expert_num,
            global_weight=None,
            full_router_logits=full_router_logits,
            routing_weights=routing_weights,
            aux_balance_weight=aux_balance_weight,
        )

        #
        # ---------------- token drop ----------------
        #

        if self.token_drop:  # and self.training:
            expert_mask_dtype = expert_mask.dtype
            capacity = _capacity(batch_size * sequence_length, self.mlp_dynamic_expert_num, torch.tensor(self.capacity_factor), torch.tensor(self.min_capacity))
            if self.drop_policy == "probs":
                if capacity > dynamic_router_logits.shape[0]:
                    print(f"[warning] token capacity({capacity}) > token num({dynamic_router_logits.shape[0]}), setting capacity=token num")
                    capacity = dynamic_router_logits.shape[0]
                dynamic_expert_mask = expert_mask[:, : self.mlp_dynamic_expert_num].bool()
                token_drop_router_logits = torch.masked_fill(dynamic_router_logits, ~dynamic_expert_mask, torch.finfo(dynamic_router_logits.dtype).min)
                capacity_probs, capacity_indices = torch.topk(token_drop_router_logits, k=capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(expert_mask).scatter(0, capacity_indices, 1)
                capacity_mask[:, self.mlp_dynamic_expert_num :] = 1
                expert_mask = torch.logical_and(expert_mask, capacity_mask)

                ori_token_num = dynamic_expert_mask.sum().item()
                cur_token_num = expert_mask[:, : self.mlp_dynamic_expert_num].sum().item()
                if self.drop_token_num_print and ("RANK" not in os.environ or int(os.environ["RANK"]) == 0):
                    print(f"drop {ori_token_num - cur_token_num} tokens from total {ori_token_num} tokens")

            elif self.drop_policy == "position":
                locations = torch.cumsum(expert_mask, dim=0) - 1
                expert_mask *= torch.lt(locations, capacity)
            else:
                raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
            expert_mask = expert_mask.to(expert_mask_dtype)

            # V5 tips: 这里修改了expert mask, 所以routing_weights有些有权重的地方需要重新mask, 然后重新unify一次, 相对比例不变
            routing_weights = routing_weights.masked_fill(~(expert_mask[:, : self.mlp_dynamic_expert_num].bool()), 0.0)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1).unsqueeze(-1).expand(-1, routing_weights.shape[-1]) + 1e-6)

        # global_weight 存储全局权重分配
        if self.mlp_dynamic_expert_num < self.num_experts:
            global_weight = cal_global_weight(expert_mask, full_router_logits, self.mlp_dynamic_expert_num, routing_weights)
        else:
            global_weight = routing_weights

        # 懒得看之前的代码了, 这里放一个debug
        # if not ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).sum() == 0:
        #     iindex = ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).nonzero().tolist()[:1]
        #     xxx  = (expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)
        #     raise ValueError(f"iindex: {iindex}\nexpert_mask: {[expert_mask[xx[0]] for xx in iindex]}\nattention_mask: {[attention_mask[xx[0]] for xx in iindex]}\nglobal_weight: {[global_weight[xx[0]] for xx in iindex]}\nrouting_weights: {[routing_weights[xx[0]] for xx in iindex]}\nxxx: {[xxx[xx[0]] for xx in iindex]}")
        assert ((expert_mask == 0) & (attention_mask == 1) & (global_weight != 0)).sum() == 0
        # assert ((expert_mask != 0) & (global_weight <= 0)).sum() == 0 # global_weight有时候还会是0, 主要原因是routing_weights的不确定性
        # assert (global_weight.sum(dim=1) != 1).sum() == 0 # 精度差异

        #
        # ---------------- 路由计算结束, 开始过expert ----------------
        #

        hidden_states = original_hidden_states.view(-1, hidden_dim)

        #  final_hidden_states: moe最终输出的表示; expert_mask:moe的路由情况 (batch * sequence_length, expert_num)
        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        global_weight = global_weight.to(hidden_states.dtype)

        #
        # ---------------- 空专家 (旧代码) ----------------
        #

        # 空专家不需要专家并行, 对最终结果也没有影响, 这里直接跳过就好了-_-
        # for expert_idx in range(self.mlp_dynamic_real_expert_num, self.mlp_dynamic_expert_num):
        #     expert_layer = self.dynamic_null_experts[expert_idx - self.mlp_dynamic_real_expert_num]
        #     top_x = torch.nonzero(expert_mask[:, expert_idx], as_tuple=True)[0]

        #     if top_x.shape[0] == 0:
        #         continue

        #     # in torch it is faster to index using lists than torch tensors
        #     top_x_list = top_x.tolist()

        #     # Index the correct hidden states and compute the expert hidden state for
        #     # the current expert. We need to make sure to multiply the output hidden
        #     # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        #     current_state = hidden_states[top_x_list]
        #     current_hidden_states = expert_layer(current_state) * global_weight[top_x_list, None, expert_idx]

        #     # However `index_add_` only support torch tensors for indexing so we'll use
        #     # the `top_x` tensor here.
        #     final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        #
        # ---------------- 动态实专家 ----------------
        #

        current_hidden_states = self.dynamic_real_moe(hidden_states, expert_mask=expert_mask[:, : self.mlp_dynamic_real_expert_num], router_weight=global_weight[:, : self.mlp_dynamic_real_expert_num])
        final_hidden_states = final_hidden_states + current_hidden_states

        #
        # ---------------- 固定专家 ----------------
        #

        for expert_idx in range(self.mlp_fixed_expert_num):
            expert_layer = self.fixed_real_moe[expert_idx]

            current_state = hidden_states
            current_global_weight = global_weight[:, self.mlp_dynamic_expert_num + expert_idx].unsqueeze(-1)  # fixed expert的weight
            current_hidden_states = expert_layer(current_state) * current_global_weight

            final_hidden_states = final_hidden_states + current_hidden_states

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if not self.training and self.avg_hidden_states_last:
            # Todo: 不知道为什么多卡之间会出现hidden state不同步的情况
            # 目前是在推理的时候发现的, 相同的数据在EP4的情况下, 第18层hidden state会出现第一次不同, 个人认为是精度和卡不同导致的小误差, 但是在之后的层数会被放大
            # https://docs.pytorch.org/docs/stable/distrisbuted.html#torch.distributed.ReduceOp
            dist.all_reduce(final_hidden_states, op=dist.ReduceOp.AVG, group=self.dynamic_real_moe.deepspeed_moe.ep_group)

        return final_hidden_states, full_router_logits, dynamic_top_k, expert_mask, global_weight, aux_loss


def load_balancing_loss_func(
    expert_mask: torch.Tensor,
    mlp_dynamic_expert_num: int,
    global_weight: Optional[torch.Tensor] = None,
    full_router_logits: Optional[torch.Tensor] = None,
    routing_weights: Optional[torch.Tensor] = None,
    aux_balance_weight: Optional[torch.Tensor] = None,
    version=2, # 初期实验证明version2好一些
) -> float:
    # Todo: top K的专家数目K会影响aux balance loss吗? aux balance loss会不会影响top P的路由的选择数目的情况?
    if version == 1:
        assert False
        # 这个函数认为前 mlp_dynamic_expert_num 是dynamic expert, 后面都是shared expert
        # 并且将固定计算所有token的路由, 尝试解决shared expert的tokens_per_expert固定为1的缺点 (solution flag)

        # 根据full_router_logits计算global weight
        if global_weight is None:
            global_weight = cal_global_weight(expert_mask, full_router_logits, mlp_dynamic_expert_num, routing_weights)

        num_experts = expert_mask.shape[1]
        if aux_balance_weight is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # solution flag
            tokens_per_expert[mlp_dynamic_expert_num:] = torch.mean(tokens_per_expert[:mlp_dynamic_expert_num])
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(global_weight, dim=0)
        else:
            batch_size, sequence_length = aux_balance_weight.shape
            num_hidden_layers = global_weight.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = aux_balance_weight[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(global_weight.device)

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
            # solution flag
            tokens_per_expert[mlp_dynamic_expert_num:] = torch.mean(tokens_per_expert[:mlp_dynamic_expert_num])

            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(global_weight * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

        return overall_loss * num_experts

    elif version == 2:
        # 这个函数认为前 mlp_dynamic_expert_num 是dynamic expert, 后面都是shared expert
        # 并且将固定计算dynamic_expert token的路由

        min_dtype = torch.finfo(full_router_logits.dtype).min  # 防止expert_mask 全0的时候出现nan, 所以不用-inf
        global_weight = full_router_logits.masked_fill(expert_mask == 0, min_dtype)
        global_weight = global_weight[:, :mlp_dynamic_expert_num]
        global_weight = torch.softmax(global_weight, dim=-1)
        expert_mask = expert_mask[:, :mlp_dynamic_expert_num]

        num_experts = expert_mask.shape[-1]
        if aux_balance_weight is None:
            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.mean(global_weight, dim=0)
        else:
            batch_size, sequence_length = aux_balance_weight.shape
            num_hidden_layers = global_weight.shape[0] // (batch_size * sequence_length)

            # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
            expert_attention_mask = aux_balance_weight[None, :, :, None].expand((num_hidden_layers, batch_size, sequence_length, num_experts)).reshape(-1, num_experts).to(global_weight.device)

            # Compute the percentage of tokens routed to each experts
            tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
            # Compute the average probability of routing to these experts
            router_prob_per_expert = torch.sum(global_weight * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)

        return overall_loss * num_experts

    else:
        raise KeyError


#
# ---------------------------- deepspeed moe ----------------------------
#


# 没有改
class Experts(deepspeed.moe.experts.Experts):
    def __init__(self, expert, num_local_experts=1, expert_group_name=None):
        super(deepspeed.moe.experts.Experts, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList([copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.deepspeed_experts:
            # TODO: Create param groups to handle expert + data case (e.g. param.group = moe_group)
            for name, param in expert.named_parameters():
                param.allreduce = False
                param.group_name = expert_group_name

    def forward(self, inputs):
        chunks = inputs.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = expert(chunk)
            if type(out) is tuple:
                out = out[0]  # Ignore the bias term for now
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=1)
        return expert_output


class MOELayer(deepspeed.moe.sharded_moe.MOELayer):
    def __init__(
        self,
        experts: nn.Module,
        ep_group_name,
        ep_size,
        num_local_experts: int,
        # use_tutel: bool = False # 强制False, 后面的逻辑都删了
    ) -> None:
        super(deepspeed.moe.sharded_moe.MOELayer, self).__init__()

        self.experts = experts
        self.ep_group = None
        self.ep_size = ep_size
        self.ep_group_name = ep_group_name
        self.num_local_experts = num_local_experts
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False

    # copy
    def _set_ep_group(self, ep_group):
        self.ep_group = ep_group

    def forward(self, hidden_states: Tensor, expert_mask: Tensor, router_weight: Tensor) -> Tensor:
        # hidden_states: [B * S, d_model]
        # expert_mask: [B * S, expert_num]
        # router_weight: [B * S, expert_num], 注意这里router_weight和不一定是0, 空专家和fix专家的没放上去

        router_weight = router_weight * expert_mask

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = hidden_states.shape[-1]
        seq_len = hidden_states.shape[0]
        expert_num = expert_mask.shape[-1]

        # motified from deepspeed topK gating, 最长capacity的同步, 为了通讯能够shape对齐
        # Communicate across expert processes to pick the maximum capacity.
        capacity = expert_mask.sum(dim=0).max()
        if self.ep_group is not None:
            dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self.ep_group)
        # # 这里还没看, 但是暂时用不上, 应该和token drop相关
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
        #     # This is since we are going to activate drop_tokens() to drop duplicate tokens.
        #     tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
        #     new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)

        # # Todo 两次compress_matrix可加速
        compres_hidden_states = hidden_states.unsqueeze(1).expand(seq_len, expert_num, d_model)  # [B * S, expert_num, d_model]
        compres_hidden_states = compress_matrix(compres_hidden_states, expert_mask, force_dim=capacity, allow_larger_dim=True)  # [C, expert_num, d_model]
        compres_expert_mask = compress_matrix(expert_mask, expert_mask, force_dim=capacity, allow_larger_dim=True)
        dispatched_input = einsum("ce,cem->ecm", compres_expert_mask, compres_hidden_states)

        # # 这里先不考虑capacity, 将所有序列的token都all reduce, 这样通讯量会变大, 后续改良,
        # # 此外, 即使没有路由的token的hidden_state也会作为全0向量输入到expert中, 输出的时候记得再过一遍 expert mask (不能保证expert的输入为0时输出一定为0)
        # dispatched_input = einsum("se,sm->esm", expert_mask, hidden_states)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()

        # 不考虑 tensor-parallel 的情况: Todo
        assert deepspeed.utils.groups.mpu is None
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # If the non-expert is tensor-parallel, it will create
        #     # duplicate tokens on the tensor-parallel ranks.
        #     # Since our experts are not tensor-parallel, these duplicates
        #     # need to be dropped to ensure correctness.
        #     # this also doubles up as a communication optimization as we are
        #     # reducing the all-to-all communication volume.
        #     # raise NotImplementedError
        #     dispatched_input = drop_tokens(dispatched_input, dim=1)

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)

        expert_output = self.experts(dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)

        # 不考虑 tensor-parallel 的情况
        # if groups._get_expert_model_parallel_world_size() == 1:
        #     # the dropped duplicate tokens need to be gathered on each
        #     # tensor parallel rank again for the tensor-parallel
        #     # non-expert of the next layer.
        #     # raise NotImplementedError
        #     expert_output = gather_tokens(expert_output, dim=1)

        # 使用压缩 ecm -> sem
        expert_output = decompress_matrix(expert_output.transpose(0, 1), expert_mask, allow_larger_dim=True)  # [B * S, expert_num, d_model]
        combined_output = einsum("se,sem->sm", router_weight, expert_output)

        # # 这里router_weight在之前用点乘融合了expert_mask, 确保了输入为0的地方输出也为0了
        # combined_output = einsum("se,esm->sm", router_weight, expert_output)

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return combined_output


class MoE(deepspeed.moe.layer.MoE):
    def __init__(self, config, expert, num_experts, ep_size, moe_name_prefix="ep_size"):
        super(deepspeed.moe.layer.MoE, self).__init__()

        # self.use_residual = use_residual # 强制False, 后面的逻辑都删了

        self.enable_expert_tensor_parallelism = config.enable_expert_tensor_parallelism
        self.ep_size = ep_size
        self.num_experts = num_experts

        assert self.num_experts % self.ep_size == 0, f"Number of experts ({self.num_experts}) should be divisible by expert parallel size ({self.ep_size})"

        self.expert_group_name = f"{moe_name_prefix}_{self.ep_size}"
        self.num_local_experts = self.num_experts // self.ep_size

        log_dist(f"Creating MoE layer with num_experts: {self.num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}", [0])

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(experts, self.expert_group_name, self.ep_size, self.num_local_experts)

    # copy
    def set_deepspeed_parallelism(self, use_data_before_expert_parallel_=False):
        self._create_process_groups(use_data_before_expert_parallel_=use_data_before_expert_parallel_)

    # copy
    def _create_process_groups(self, use_data_before_expert_parallel_=False):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu, use_data_before_expert_parallel_=use_data_before_expert_parallel_)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, *input_args, **input_kwargs):
        return self.deepspeed_moe(*input_args, **input_kwargs)


#
# ---------------------------- moe done ----------------------------
#

