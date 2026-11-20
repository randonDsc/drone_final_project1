"""Continual Backprop for stable-baselines3 SAC.

Dohare et al., "Loss of plasticity in deep continual learning", Nature 2024.
Settings and the rule are in path_sac_config.CONTINUAL_BACKPROP. In short, after
every gradient update, for each hidden layer of the actor and both critics:

    utility = decay * utility + (1 - decay) * mean|h| * sum|outgoing weights|

A neuron older than maturity updates is eligible, and the eligible neurons with
the lowest utility are replaced at replacement_rate: incoming weights and bias
redrawn as PyTorch initializes nn.Linear, outgoing weights zeroed (so the new
neuron changes nothing at first), Adam's moments for those weights zeroed, and
its age and utility reset. The target critics get the same reset, so the critic
and its slowly following copy keep describing the same neurons.

mean|h| comes from the batches of the gradient update itself: forward hooks on
each activation record only while ContinualBackpropSAC.train runs, never while
the policy flies.

Usage, after SAC is created or loaded:

    model = ContinualBackpropSAC(...)           # or ContinualBackpropSAC.load(...)
    model.attach_continual_backprop(settings)

Ages and utilities are not saved with the model. After a resume they start at
zero, so every neuron is protected again for its first maturity updates. A model
saved by ContinualBackpropSAC loads with plain SAC.load for evaluation and flight.
"""
import math

import torch
from torch import nn
from stable_baselines3 import SAC


def hidden_layers(network, heads=()):
    """(incoming Linear, activation, outgoing Linears) for each hidden layer of
    an SB3 MLP. heads read the last hidden layer when the network itself ends
    with an activation, as the actor's latent_pi does."""
    modules = list(network)
    layers = []

    for index, module in enumerate(modules[:-1]):
        if not isinstance(module, nn.Linear):
            continue

        activation = modules[index + 1]

        if isinstance(activation, nn.Linear):
            continue

        following = [m for m in modules[index + 2:] if isinstance(m, nn.Linear)]
        outgoing = following[:1] if following else list(heads)

        if not outgoing:
            raise ValueError("A hidden layer has nothing reading it.")

        layers.append((module, activation, outgoing))

    return layers


def _zero_adam_state(optimizer, parameter, index, dim):
    """Zero Adam's moment estimates for the given rows (dim 0) or columns (1)."""
    state = optimizer.state.get(parameter)

    if not state:
        return

    for key in ("exp_avg", "exp_avg_sq"):
        if key in state:
            state[key].index_fill_(dim, index, 0.0)


class Layer:
    """One hidden layer's utility tracking and neuron replacement."""

    def __init__(self, name, incoming, activation, outgoing, optimizer, mirror):
        self.name = name
        self.incoming = incoming
        self.outgoing = outgoing
        self.optimizer = optimizer

        # (incoming, outgoing) in the target network, or None.
        self.mirror = mirror

        units = incoming.out_features
        device = incoming.weight.device

        self.utility = torch.zeros(units, device=device)
        self.age = torch.zeros(units, dtype=torch.long, device=device)
        self.pending = 0.0
        self.replaced = 0

        self.activation_sum = torch.zeros(units, device=device)
        self.batches = 0
        self.mean_activation = torch.zeros(units, device=device)

        self.recording = False
        activation.register_forward_hook(self._record)

    def _record(self, module, inputs, output):
        if not self.recording:
            return

        values = output.detach().abs().reshape(-1, output.shape[-1])
        self.activation_sum += values.mean(dim=0)
        self.batches += 1

    def step(self, updates, settings):
        """Advance by `updates` gradient updates; returns how many neurons were
        replaced."""
        self.age += updates

        if self.batches == 0:
            return 0

        self.mean_activation = self.activation_sum / self.batches
        self.activation_sum.zero_()
        self.batches = 0

        with torch.no_grad():
            outgoing = sum(
                layer.weight.abs().sum(dim=0) for layer in self.outgoing
            )
            contribution = self.mean_activation * outgoing

            decay = settings["decay"] ** updates
            self.utility.mul_(decay).add_((1.0 - decay) * contribution)

        eligible = torch.nonzero(self.age > settings["maturity"]).flatten()

        self.pending += settings["replacement_rate"] * len(eligible) * updates
        count = min(int(self.pending), len(eligible))

        if count == 0:
            return 0

        self.pending -= count

        lowest = torch.topk(self.utility[eligible], count, largest=False).indices
        self.reset(eligible[lowest], settings)

        return count

    @torch.no_grad()
    def reset(self, index, settings):
        incoming = self.incoming

        # nn.Linear's own initialization, drawn for the whole layer so the
        # fan-in matches, then only the chosen rows kept.
        fresh = torch.empty_like(incoming.weight)
        nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
        incoming.weight[index] = fresh[index]

        if incoming.bias is not None:
            bound = 1.0 / math.sqrt(incoming.in_features)
            incoming.bias[index] = torch.empty(
                len(index), device=incoming.bias.device
            ).uniform_(-bound, bound)

        mean = self.mean_activation[index]

        for layer in self.outgoing:
            if settings["bias_correction"] and layer.bias is not None:
                layer.bias += layer.weight[:, index] @ mean

            layer.weight[:, index] = 0.0

        _zero_adam_state(self.optimizer, incoming.weight, index, 0)

        if incoming.bias is not None:
            _zero_adam_state(self.optimizer, incoming.bias, index, 0)

        for layer in self.outgoing:
            _zero_adam_state(self.optimizer, layer.weight, index, 1)

        if self.mirror is not None:
            target_incoming, target_outgoing = self.mirror
            target_incoming.weight[index] = incoming.weight[index]

            if incoming.bias is not None:
                target_incoming.bias[index] = incoming.bias[index]

            for layer in target_outgoing:
                if settings["bias_correction"] and layer.bias is not None:
                    layer.bias += layer.weight[:, index] @ mean

                layer.weight[:, index] = 0.0

        self.utility[index] = 0.0
        self.age[index] = 0
        self.replaced += len(index)


class ContinualBackprop:
    """Every hidden layer of a SAC model's actor and critics."""

    def __init__(self, model, settings):
        if model.use_sde:
            raise ValueError("Continual Backprop here does not support gSDE.")

        self.settings = dict(settings)
        self.layers = []

        actor = model.actor

        for number, (incoming, activation, outgoing) in enumerate(
            hidden_layers(actor.latent_pi, heads=(actor.mu, actor.log_std))
        ):
            self.layers.append(Layer(
                f"actor_{number}", incoming, activation, outgoing,
                actor.optimizer, mirror=None,
            ))

        pairs = zip(model.critic.q_networks, model.critic_target.q_networks)

        for critic_number, (q_network, target_network) in enumerate(pairs):
            online = hidden_layers(q_network)
            target = hidden_layers(target_network)

            for number, (incoming, activation, outgoing) in enumerate(online):
                target_incoming, _, target_outgoing = target[number]

                self.layers.append(Layer(
                    f"critic{critic_number}_{number}",
                    incoming, activation, outgoing,
                    model.critic.optimizer,
                    mirror=(target_incoming, target_outgoing),
                ))

    def recording(self, on):
        for layer in self.layers:
            layer.recording = on

    def step(self, updates):
        return sum(layer.step(updates, self.settings) for layer in self.layers)

    def statistics(self):
        """For the training log: replacements so far and, from the last update,
        the share of neurons that never activated on its batches."""
        result = {}

        for group in ("actor", "critic"):
            layers = [l for l in self.layers if l.name.startswith(group)]
            units = sum(len(l.mean_activation) for l in layers)
            silent = sum(int((l.mean_activation == 0).sum()) for l in layers)

            result[f"cbp/{group}_replaced"] = sum(l.replaced for l in layers)
            result[f"cbp/{group}_silent_share"] = silent / max(units, 1)

        return result


class ContinualBackpropSAC(SAC):
    """SAC that runs Continual Backprop after each training call. Without
    attach_continual_backprop it trains exactly as SAC does."""

    continual_backprop = None

    def attach_continual_backprop(self, settings):
        self.continual_backprop = ContinualBackprop(self, settings)

    def _excluded_save_params(self):
        # Holds hooks and live tensors; it is rebuilt by attaching again.
        return super()._excluded_save_params() + ["continual_backprop"]

    def train(self, gradient_steps, batch_size=64):
        cbp = self.continual_backprop

        if cbp is None:
            return super().train(gradient_steps, batch_size)

        cbp.recording(True)

        try:
            super().train(gradient_steps, batch_size)
        finally:
            cbp.recording(False)

        cbp.step(gradient_steps)

        for key, value in cbp.statistics().items():
            self.logger.record(key, value)
