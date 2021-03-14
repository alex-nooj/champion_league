from typing import Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from adept.utils.util import DotDict
from torch.utils.data import BatchSampler, SequentialSampler
import numpy as np

from champion_league.agent.base.base_agent import Agent
from champion_league.utils.rewardnorms import ScaleNorm
from champion_league.utils.rollout import Rollout


class PPOAgent(Agent):
    def __init__(
            self,
            args: DotDict,
            rollout_len: int,
            batch_size: int,
            discount: float,
            entropy_weight: float,
            gae_discount: float,
            policy_clipping: float,
            device: int,
            nb_rollout_epoch: int,
            rollout_minibatch_len: int
    ):
        super().__init__(args)
        self._args = args
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.discount = discount
        self.entropy_weight = entropy_weight
        self.gae_discount = gae_discount
        self.policy_clipping = policy_clipping
        self.nb_rollout_epoch = nb_rollout_epoch
        self.rollout_minibatch_len = rollout_minibatch_len
        self.device = device
        self.memory = Rollout(rollout_len, batch_size)
        self.reward_normalizer = ScaleNorm(11)

    def learn_step(self, optimizer, network, next_state, next_internals) -> Dict[str, float]:
        torch.autograd.set_detect_anomaly(True)
        network.train()
        # Read the experience replay
        r = self.memory.read()
        device = r.rewards[0].device
        rollout_len = self.memory.rollout_len

        with torch.no_grad():
            pred, _ = network(next_state.squeeze(), next_internals)
            last_values = pred["critic"].squeeze(-1).data

        # Calculate n-steps
        gae = 0.0
        next_values = last_values
        gae_returns = []
        for i in reversed(range(rollout_len)):
            rewards = self.reward_normalizer(r.rewards[i])
            terminal_mask = 1. - r.terminals[i].float()
            current_values = r.values[i].squeeze(-1)

            # Generalized advantage estimation
            delta_t = rewards + self.discount * next_values.data * terminal_mask - current_values
            gae = gae * self.discount * self.gae_discount * terminal_mask + delta_t
            gae_returns.append(gae + current_values)
            next_values = current_values.data
        gae_returns = torch.stack(list(reversed(gae_returns))).data

        # Convert to torch tensors of [seq, batch_size]
        old_values = r.values.squeeze(-1)
        adv_targets_batch = (gae_returns - old_values).data
        old_log_probs_batch = r.log_probs.data
        rollout_terminals = r.terminals.cpu().numpy()

        # # Normalize advantage
        # if self.normalize_advantage:
        #     adv_targets_batch = (adv_targets_batch - adv_targets_batch.mean()) / (adv_targets_batch.std() + 1e-5)

        metric_loss = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "entropy_loss": 0.0
        }

        for e in range(self.nb_rollout_epoch):
            # setup minibatch iterator
            minibatch_inds = list(BatchSampler(
                SequentialSampler(range(rollout_len)),
                self.rollout_minibatch_len, drop_last=False)
            )
            # randomize sequences to sample NOTE: in-place operation
            np.random.shuffle(minibatch_inds)
            for i in minibatch_inds:
                # starting_internals needs to be a dict with "hx" and "cx", each being a tuple of
                # size batch_size containing the internals of the network
                starting_internals = {
                    "hx": r.hx[i[0]].squeeze(),
                    "cx": r.cx[i[0]].squeeze()
                }
                gae_return = gae_returns[i]
                old_log_probs = old_log_probs_batch[i]
                sampled_actions = [r.actions[x] for x in i]
                batch_states = [r.states[x] for x in i]
                adv_targets = adv_targets_batch[i].unsqueeze(-1)
                terminals_batch = rollout_terminals[i]

                # forward pass
                cur_log_probs, cur_values, entropies = self.act_batch(
                    network, batch_states, terminals_batch, sampled_actions, starting_internals, device
                )
                value_loss = 0.5 * torch.mean((cur_values - gae_return).pow(2))

                # Calculate surrogate loss
                # [32, 64, 1] [32, 64, 1, 22]
                surrogate_ratio = torch.exp(cur_log_probs - old_log_probs.unsqueeze(-1))
                surrogate_loss = surrogate_ratio * adv_targets
                surrogate_loss_clipped = torch.clamp(
                    surrogate_ratio,
                    min=1 - self.policy_clipping,
                    max=1 + self.policy_clipping
                ) * adv_targets
                policy_loss = torch.mean(-torch.min(surrogate_loss, surrogate_loss_clipped))
                entropy_loss = torch.mean(-self.entropy_weight * entropies)

                losses = {
                    "value_loss": value_loss,
                    "policy_loss": policy_loss,
                    "entropy_loss": entropy_loss
                }

                for key in metric_loss:
                    metric_loss[key] += losses[key]
                total_loss = torch.sum(torch.stack(
                    tuple(loss for loss in losses.values())
                ))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        network.eval()
        return metric_loss

    def act_batch(self, network, batch_obs, batch_terminals, batch_actions, internals, device):
        exp_cache = {
            "log_probs": [],
            "entropies": [],
            "values": []
        }

        for obs, actions, terminals in zip(batch_obs, batch_actions, batch_terminals):
            preds, internals = network(obs, internals)
            for key, v in self._process_exp(preds, actions).items():
                exp_cache[key].append(v)

            terminal_inds = np.where(terminals)[0]
            for i in terminal_inds:
                for k, v in network.new_internals().items():
                    internals[k][i] = v

        return torch.stack(exp_cache["log_probs"]), torch.stack(exp_cache["values"]), torch.stack(exp_cache["entropies"])

    @staticmethod
    def _process_exp(preds, sampled_actions):
        value = preds["critic"].squeeze(1)

        logit = preds["action"].view(preds["action"].shape[0], -1)

        log_softmax = F.log_softmax(logit, dim=1)
        softmax = F.softmax(logit, dim=1)
        entropy = -(log_softmax * softmax).sum(1, keepdim=True)

        log_prob = log_softmax.gather(1, sampled_actions.unsqueeze(1))

        return {
            "log_probs": log_prob,
            "entropies": entropy,
            "values": value
        }

    @staticmethod
    def choose_move(action_probs: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(action_probs, 1)

    def reset(self) -> None:
        self.memory.clear()
