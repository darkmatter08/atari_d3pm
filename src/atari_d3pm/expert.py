"""Pinned CleanRL Pong expert and its evaluation observation pipeline."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np


EXPERT_REPOSITORY = (
    "cleanrl/Pong-v5-cleanba_ppo_envpool_impala_atari_wrapper-seed1"
)
EXPERT_REVISION = "f2ad2531c78cc639f2a54511aa8716765c33499d"
EXPERT_FILENAME = "cleanba_ppo_envpool_impala_atari_wrapper.cleanrl_model"
EXPERT_SHA256 = "7b76801eff3153a6f55a87c2a6e221d96b6a428fb76c2c29fdedc93f204a96e4"
EXPERT_URL = (
    f"https://huggingface.co/{EXPERT_REPOSITORY}/resolve/"
    f"{EXPERT_REVISION}/{EXPERT_FILENAME}"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_expert(path: str | Path) -> Path:
    """Download the immutable expert artifact and verify its content hash."""
    path = Path(path)
    if path.exists():
        actual = sha256(path)
        if actual != EXPERT_SHA256:
            raise RuntimeError(f"Expert hash mismatch: {actual} != {EXPERT_SHA256}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(EXPERT_URL, temporary)
    actual = sha256(temporary)
    if actual != EXPERT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded expert hash mismatch: {actual} != {EXPERT_SHA256}")
    temporary.replace(path)
    return path


def make_expert_env():
    """Build the Gymnasium equivalent of the expert's EnvPool Atari pipeline."""
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

    gym.register_envs(ale_py)
    base = gym.make(
        "ALE/Pong-v5",
        obs_type="rgb",
        frameskip=1,
        repeat_action_probability=0.0,
        full_action_space=False,
        max_episode_steps=108_000,
    )
    env = AtariPreprocessing(
        base,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4, padding_type="reset")
    if env.observation_space.shape != (4, 84, 84):
        env.close()
        raise RuntimeError(f"Unexpected expert observation space: {env.observation_space}")
    meanings = list(env.unwrapped.get_action_meanings())
    expected = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    if meanings != expected:
        env.close()
        raise RuntimeError(f"Unexpected Pong action mapping: {meanings}")
    return env


def current_rgb_frame(env) -> np.ndarray:
    """Read the raw ALE screen at the current decision point."""
    frame = np.asarray(env.unwrapped.ale.getScreenRGB(), dtype=np.uint8)
    if frame.shape != (210, 160, 3):
        raise RuntimeError(f"Unexpected raw ALE frame shape: {frame.shape}")
    return frame.copy()


def _model_types():
    import flax.linen as nn
    import jax.numpy as jnp
    from flax.linen.initializers import constant, orthogonal

    class ResidualBlock(nn.Module):
        channels: int

        @nn.compact
        def __call__(self, x):
            inputs = x
            x = nn.relu(x)
            x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
            return x + inputs

    class ConvSequence(nn.Module):
        channels: int

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(self.channels, kernel_size=(3, 3))(x)
            x = nn.max_pool(
                x, window_shape=(3, 3), strides=(2, 2), padding="SAME"
            )
            x = ResidualBlock(self.channels)(x)
            return ResidualBlock(self.channels)(x)

    class Network(nn.Module):
        channelss: tuple[int, ...] = (16, 32, 32)

        @nn.compact
        def __call__(self, x):
            x = jnp.transpose(x, (0, 2, 3, 1)) / 255.0
            for channels in self.channelss:
                x = ConvSequence(channels)(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(
                256,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            return nn.relu(x)

    class Actor(nn.Module):
        action_dim: int

        @nn.compact
        def __call__(self, x):
            return nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(x)

    class Critic(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(
                1, kernel_init=orthogonal(1), bias_init=constant(0.0)
            )(x)

    return Network, Actor, Critic


class CleanRLPongExpert:
    """Load and run the version-pinned Flax CleanRL policy."""

    def __init__(self, checkpoint: str | Path, mode: str = "deterministic") -> None:
        if mode not in {"deterministic", "stochastic"}:
            raise ValueError("mode must be 'deterministic' or 'stochastic'")
        checkpoint = Path(checkpoint)
        if sha256(checkpoint) != EXPERT_SHA256:
            raise RuntimeError("Refusing to load an unverified expert checkpoint")

        import flax.serialization
        import jax
        import jax.numpy as jnp

        Network, Actor, Critic = _model_types()
        network = Network()
        actor = Actor(action_dim=6)
        critic = Critic()
        key = jax.random.PRNGKey(1)
        key, network_key, actor_key, critic_key = jax.random.split(key, 4)
        example = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        network_params = network.init(network_key, example)
        hidden = network.apply(network_params, example)
        actor_params = actor.init(actor_key, hidden)
        critic_params = critic.init(critic_key, hidden)
        with checkpoint.open("rb") as stream:
            _, parameters = flax.serialization.from_bytes(
                (None, (network_params, actor_params, critic_params)), stream.read()
            )
        network_params, actor_params, _ = parameters

        def logits(observations):
            features = network.apply(network_params, observations)
            return actor.apply(actor_params, features)

        self._jax = jax
        self._jnp = jnp
        self._logits = jax.jit(logits)
        self._key = jax.random.PRNGKey(1)
        self.mode = mode

    def actions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.uint8)
        if observations.ndim == 3:
            observations = observations[None]
        if observations.ndim != 4 or observations.shape[1:] != (4, 84, 84):
            raise ValueError(f"Expected [B, 4, 84, 84], got {observations.shape}")
        logits = self._logits(observations)
        if self.mode == "deterministic":
            result = self._jnp.argmax(logits, axis=-1)
        else:
            self._key, subkey = self._jax.random.split(self._key)
            result = self._jax.random.categorical(subkey, logits, axis=-1)
        return np.asarray(result, dtype=np.int64)

    def action(self, observation: np.ndarray) -> int:
        return int(self.actions(observation)[0])
