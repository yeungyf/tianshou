"""
DQN (Deep Q-Network) 算法实现 - 详细注释版

═══════════════════════════════════════════════════════════════════════════════
算法概览
═══════════════════════════════════════════════════════════════════════════════

DQN 是一种 Off-Policy 的 Q-learning 算法，用深度神经网络近似 Q(s,a)，解决了
高维状态/动作空间的表格 Q-learning 不可行的问题。

核心思想：
- 用神经网络拟合 Q 函数：Q(s,a) ≈ Q_θ(s,a)
- Replay Buffer 存储经验 (s,a,r,s')，打破数据相关性
- Target Network 定期同步，稳定 TD 目标
- epsilon-greedy 探索：以 ε 概率随机动作，否则贪心

═══════════════════════════════════════════════════════════════════════════════
DQN vs PPO 对比
═══════════════════════════════════════════════════════════════════════════════

特性              | DQN (Off-Policy)        | PPO (On-Policy)
-----------------|-------------------------|---------------------------
学习方式          | Q-learning              | Policy Gradient
数据使用          | Replay Buffer (重复使用) | 当前策略收集，用完即弃
适用场景          | 离散动作空间             | 连续/离散动作空间
探索方式          | Epsilon-greedy          | 策略本身的随机性
目标网络          | 需要                    | 不需要
样本效率          | 高                      | 相对较低

═══════════════════════════════════════════════════════════════════════════════
执行流程 Chain
═══════════════════════════════════════════════════════════════════════════════

训练一个 epoch 的流程：
1. 【数据收集】Collector 用当前策略（epsilon-greedy）收集 N 步数据 → Replay Buffer
2. 【采样】从 Buffer 中采样 mini-batch (s,a,r,s',...)
3. 【预处理】_preprocess_batch:
   - 计算 n-step return：y = r + γr' + ... + γ^{n-1} r_{n-1} + γ^n max_a' Q_target(s',a')
   - 得到 batch.returns 作为 TD 目标
4. 【更新】_update_with_batch:
   a) 按 target_update_freq 决定是否同步 target 网络
   b) 当前 Q(s,a) = policy(batch).logits 取对应动作的 Q
   c) TD error: δ = returns - Q(s,a)
   d) Loss: MSE(δ) 或 Huber(δ)，可选 weight（优先经验回放）
   e) 反向传播，更新 Q 网络
5. 数据保留在 Buffer 中，下次继续采样使用

关键数据流：
Environment → Collector → ReplayBuffer → DQN.preprocess (n-step return)
                                      ↓
                            DQN.update (TD loss, 可选 target 更新)
                                      ↓
                            Q 网络（及可选的 target 网络）更新

═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.discrete import Discrete
from sensai.util.helper import mark_used

from tianshou.algorithm import Algorithm
from tianshou.algorithm.algorithm_base import (
    LaggedNetworkFullUpdateAlgorithmMixin,
    OffPolicyAlgorithm,
    Policy,
    TArrOrActBatch,
)
from tianshou.algorithm.modelfree.reinforce import (
    SimpleLossTrainingStats,
)
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.data.types import (
    ActBatchProtocol,
    BatchWithReturnsProtocol,
    ModelOutputBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.utils.lagged_network import EvalModeModuleWrapper
from tianshou.utils.net.common import Net

mark_used(ActBatchProtocol)  # 满足类型协议使用检查

TModel = TypeVar("TModel", bound=torch.nn.Module | Net)  # Q 网络模型类型
log = logging.getLogger(__name__)


class DiscreteQLearningPolicy(Policy, Generic[TModel]):
    """
    离散动作的 Q-learning 策略：用模型输出各动作的 Q 值，贪心选最大 Q 的动作；
    探索时通过 epsilon-greedy 以一定概率随机选动作。DQN 等算法使用本策略。
    """

    def __init__(
        self,
        *,
        model: TModel,
        action_space: gym.spaces.Space,
        observation_space: gym.Space | None = None,
        eps_training: float = 0.0,
        eps_inference: float = 0.0,
    ) -> None:
        """
        离散 Q-learning 策略初始化。

        ═══════════════════════════════════════════════════════════════════════
        关键参数说明
        ═══════════════════════════════════════════════════════════════════════

        :param model: 将 (obs, state, info) 映射为各动作 Q 值 action_values_BA 的模型
        :param action_space: 环境的动作空间（须为离散 Discrete）
        :param observation_space: 环境的观测空间（可选）
        :param eps_training: 训练时的 epsilon-greedy 探索概率
            - 收集数据时以该概率选随机动作，否则选策略给出的动作
            - 0.0 表示不探索（完全贪心），1.0 表示完全随机
        :param eps_inference: 推理/评估时的 epsilon-greedy 探索概率
            - 非训练场景（如 test 评估）下使用
            - 0.0 表示不探索，1.0 表示完全随机
        """
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=False,
            action_bound_method=None,
        )
        self.action_space = cast(Discrete, self.action_space)
        self.model = model
        self.eps_training = eps_training
        self.eps_inference = eps_inference

    def set_eps_training(self, eps: float) -> None:
        """
        设置训练时的 epsilon-greedy 探索概率。

        :param eps: 训练时以该概率选随机动作；0.0 为完全贪心，1.0 为完全随机。
        """
        self.eps_training = eps

    def set_eps_inference(self, eps: float) -> None:
        """
        设置推理/评估时的 epsilon-greedy 探索概率。

        :param eps: 推理时以该概率选随机动作；0.0 为完全贪心，1.0 为完全随机。
        """
        self.eps_inference = eps

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: Any | None = None,
        model: torch.nn.Module | None = None,
    ) -> ModelOutputBatchProtocol:
        """
        根据当前 batch 计算动作：模型输出各动作 Q 值，取 argmax 得到动作。

        若需屏蔽非法动作，请在 batch.obs 中提供 "mask"，例如三动作环境中仅动作 1 可用：
        batch.obs.mask = np.array([[False, True, False]])。

        :param batch: 观测 batch
        :param state: 可选隐状态（用于 RNN）
        :param model: 未传入时使用 self.model；通常可传入 lagged target 网络用于计算 target Q
        :return: Batch(logits=各动作 Q 值, act=动作, state=隐状态)
        """
        if model is None:
            model = self.model
        obs = batch.obs
        mask = getattr(obs, "mask", None)
        # 兼容 obs 为 Batch(obs=..., mask=...) 或 直接为数组 两种情况
        obs_arr = obs.obs if hasattr(obs, "obs") else obs
        action_values_BA, hidden_BH = model(obs_arr, state=state, info=batch.info)
        q = self.compute_q_value(action_values_BA, mask)  # 应用 mask 后得到有效 Q
        act_B = to_numpy(q.argmax(dim=1))  # 贪心：选 Q 最大的动作
        result = Batch(logits=action_values_BA, act=act_B, state=hidden_BH)
        return cast(ModelOutputBatchProtocol, result)

    def compute_q_value(self, logits: torch.Tensor, mask: np.ndarray | None) -> torch.Tensor:
        """
        根据网络原始输出与动作 mask 得到有效 Q 值：被 mask 掉的动作 Q 设为极小值，
        这样 argmax 不会选到非法动作。
        """
        if mask is not None:
            # 被 mask 掉的动作的 Q 应小于 logits.min()，确保 argmax 不选到
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def add_exploration_noise(
        self,
        act: TArrOrActBatch,
        batch: ObsBatchProtocol,
    ) -> TArrOrActBatch:
        """
        epsilon-greedy 探索：以 eps 概率将动作替换为随机动作（在合法动作内随机），
        否则保持原贪心动作。
        """
        eps = self.eps_training if self.is_within_training_step else self.eps_inference
        if np.isclose(eps, 0.0):
            return act
        if isinstance(act, np.ndarray):
            batch_size = len(act)
            rand_mask = np.random.rand(batch_size) < eps  # 哪些样本要替换为随机动作
            self.action_space = cast(Discrete, self.action_space)  # for mypy
            action_num = int(self.action_space.n)
            q = np.random.rand(batch_size, action_num)  # [0,1) 随机，用于在合法动作中 argmax
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask  # 非法动作不加分，argmax 会选合法动作
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]  # 仅对 rand_mask 为 True 的样本替换
            return act  # type: ignore[return-value]
        raise NotImplementedError(
            f"Currently only numpy array is supported for action, but got {type(act)}"
        )


TDQNPolicy = TypeVar("TDQNPolicy", bound=DiscreteQLearningPolicy)


class QLearningOffPolicyAlgorithm(
    OffPolicyAlgorithm[TDQNPolicy], LaggedNetworkFullUpdateAlgorithmMixin, ABC
):
    """
    Q-learning 类 Off-Policy 算法的基类：用 Q 函数计算 n-step return 作为 TD 目标。
    可选使用 lagged 模型作为 target network，定期与当前网络同步。
    """

    def __init__(
        self,
        *,
        policy: TDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        n_step_return_horizon: int = 1,
        target_update_freq: int = 0,
    ) -> None:
        """
        Q-learning Off-Policy 算法初始化。

        ═══════════════════════════════════════════════════════════════════════
        关键参数说明
        ═══════════════════════════════════════════════════════════════════════

        :param policy: 策略（含 Q 网络）
        :param optim: 策略/模型用的优化器工厂

        :param gamma: 折扣因子，范围 [0, 1]
            - 越接近 0 越重视即时奖励（短视），越接近 1 越重视长期回报
            - 通常取 0.9 ~ 0.99

        :param n_step_return_horizon: n-step 的步数，大于 0
            - 控制 TD 与 MC 的权衡：越大 bias 越小、variance 越大
            - 1 为单步 TD；很大时接近 MC 回报
            - 典型值 1 ~ 10

        :param target_update_freq: 每多少次训练迭代完整更新一次 target 网络
            - 0 表示不用 target 网络，只用当前网络做选动作和 bootstrap
            - 越大目标越稳定但新估计传播越慢；越小可能不稳定
            - 典型值 100 ~ 10000（视环境而定）
        """
        super().__init__(
            policy=policy,
        )
        self.optim = self._create_policy_optimizer(optim)
        LaggedNetworkFullUpdateAlgorithmMixin.__init__(self)
        assert 0.0 <= gamma <= 1.0, f"discount factor should be in [0, 1] but got: {gamma}"
        self.gamma = gamma
        assert n_step_return_horizon > 0, (
            f"n_step_return_horizon should be greater than 0 but got: {n_step_return_horizon}"
        )
        self.n_step = n_step_return_horizon
        self.target_update_freq = target_update_freq
        self._iter = 0  # 训练迭代计数，用于判断是否该更新 target
        self.model_old: EvalModeModuleWrapper | None = (
            self._add_lagged_network(self.policy.model) if self.use_target_network else None
        )

    def _create_policy_optimizer(self, optim: OptimizerFactory) -> Algorithm.Optimizer:
        """为策略的 model 创建优化器。"""
        return self._create_optimizer(self.policy, optim)

    @property
    def use_target_network(self) -> bool:
        """是否使用 target 网络（target_update_freq > 0 时使用）。"""
        return self.target_update_freq > 0

    @abstractmethod
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """由子类实现：给定 buffer 与 indices，计算 n-step 的 target Q 值（用于 TD 目标）。"""
        pass

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """
        预处理 batch：计算 n-step return 作为 Q-learning 的 TD 目标。

        详见 BasePolicy.compute_nstep_return：用 target_q_fn 在 s' 上算 target Q，
        再与 n 步奖励组合成 returns。
        """
        return self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )

    def _periodically_update_lagged_network_weights(self) -> None:
        """
        按周期更新 lagged target 网络：每 target_update_freq 次调用，将当前网络参数
        完整拷贝到 target 网络；否则只递增迭代计数。
        """
        if self.use_target_network and self._iter % self.target_update_freq == 0:
            self._update_lagged_network_weights()
        self._iter += 1


class DQN(
    QLearningOffPolicyAlgorithm[TDQNPolicy],
    Generic[TDQNPolicy],
):
    """
    Deep Q-Network 实现。论文: arXiv:1312.5602。
    支持 Double Q-Learning（arXiv:1509.06461）：用当前网络选动作、target 网络估 Q，减轻过估计。
    Dueling DQN（arXiv:1511.06581）在网络侧实现，不在此处。
    """

    def __init__(
        self,
        *,
        policy: TDQNPolicy,
        optim: OptimizerFactory,
        gamma: float = 0.99,
        n_step_return_horizon: int = 1,
        target_update_freq: int = 0,
        is_double: bool = True,
        huber_loss_delta: float | None = None,
    ) -> None:
        """
        DQN 算法初始化。

        ═══════════════════════════════════════════════════════════════════════
        关键参数说明（继承部分见 QLearningOffPolicyAlgorithm）
        ═══════════════════════════════════════════════════════════════════════

        :param policy: 策略（含 Q 网络）
        :param optim: 策略/模型用的优化器工厂
        :param gamma: 折扣因子，通常 0.9 ~ 0.99
        :param n_step_return_horizon: n-step 步数，1 为单步 TD
        :param target_update_freq: target 网络更新间隔，0 表示不用 target 网络

        --- DQN 特有参数 ---
        :param is_double: 是否使用 Double DQN 计算 target
            - True: 用当前网络选动作、target 网络估 Q，减轻过估计
            - False: 直接用 target 网络的 max Q（Nature DQN）
            - 仅当 target_update_freq > 0 时 Double 才有意义

        :param huber_loss_delta: TD 误差的损失函数
            - None: 使用 MSE
            - 非 None: 使用 Huber loss（Nature DQN 做法），delta 为阈值，限制离群点影响
            - Huber 在大误差时梯度饱和，训练更稳定；delta 大小应与回报尺度匹配
        """
        super().__init__(
            policy=policy,
            optim=optim,
            gamma=gamma,
            n_step_return_horizon=n_step_return_horizon,
            target_update_freq=target_update_freq,
        )
        self.is_double = is_double
        self.huber_loss_delta = huber_loss_delta

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """
        计算 n-step 的 target Q 值，用于 TD 目标 y = r + γ^n * target_q。
        Double DQN：用当前策略选动作 a' = argmax_a Q_new(s',a)，用 target 网络估 Q_old(s',a')。
        Nature DQN：直接用 target 网络的 max_a Q_old(s',a)，易过估计。
        """
        obs_next_batch = Batch(
            obs=buffer[indices].obs_next,
            info=[None] * len(indices),
        )  # 下一状态 s_{t+n}，用于 bootstrap
        result = self.policy(obs_next_batch)  # 用当前网络得到动作与 Q（Double 时用于选动作）
        if self.use_target_network:
            # Double DQN: target_Q = Q_old(s', a')，其中 a' = argmax_a Q_new(s', a)
            target_q = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            target_q = result.logits
        if self.is_double:
            # 取 target 网络在「当前网络所选动作」上的 Q 值
            return target_q[np.arange(len(result.act)), result.act]
        # Nature DQN：取 target 网络 max Q，易过估计
        return target_q.max(dim=1)[0]

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        """
        DQN 单步更新：按周期更新 target 网络，用 TD 目标与当前 Q 算 loss 并反向传播。

        流程：更新 target（若到周期）→ 取当前 Q(s,a) → TD error = returns - Q(s,a)
        → Loss = MSE 或 Huber（可选 weight 用于优先经验回放）→ 写回 batch.weight 供 prio-buffer → optim.step。
        """
        self._periodically_update_lagged_network_weights()  # 每 target_update_freq 次同步 target
        weight = batch.pop("weight", 1.0)  # 优先经验回放时的样本权重，默认 1.0
        q = self.policy(batch).logits  # 当前网络对各动作的 Q
        q = q[np.arange(len(q)), batch.act]  # 只取已执行动作 a 的 Q(s,a)
        returns = to_torch_as(batch.returns.flatten(), q)  # n-step TD 目标
        td_error = returns - q  # TD 误差 δ = y - Q(s,a)

        if self.huber_loss_delta is not None:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(
                y, t, delta=self.huber_loss_delta, reduction="mean"
            )
        else:
            loss = (td_error.pow(2) * weight).mean()  # MSE，带 weight 时用于 prio

        batch.weight = td_error  # 供优先经验回放：用 |δ| 做优先级
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())


# ═══════════════════════════════════════════════════════════════════════════════
# DQN 算法关键概念总结
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. TD 目标与 n-step return（核心公式）:
#    ────────────────────────────────────────
#    单步: y = r + γ max_a' Q_target(s', a')
#    n步:  y = r_0 + γ r_1 + ... + γ^{n-1} r_{n-1} + γ^n max_a' Q_target(s_n, a')
#
#    - γ 为折扣因子，n 越大 bias 越小、variance 越大
#    - target 网络用于稳定目标，避免追逐移动目标
#
# 2. Double DQN（减轻过估计）:
#    ─────────────────────────
#    Nature DQN: a' = argmax_a Q_target(s', a)  → 易过估计 max Q
#    Double DQN: a' = argmax_a Q_new(s', a),  y 中用 Q_target(s', a')
#
#    选动作用当前网络，估 Q 用 target 网络，解耦选动作与估 Q，减少过估计。
#    仅当使用 target 网络（target_update_freq > 0）时才有意义。
#
# 3. Target Network（目标网络）:
#    ──────────────────────────
#    - 每 target_update_freq 次更新，将当前 Q 网络参数完整拷贝到 target
#    - 提供相对稳定的 TD 目标，避免训练发散
#    - 典型 target_update_freq: 100 ~ 10000
#
# 4. Huber Loss vs MSE:
#    ─────────────────
#    - MSE: L = (y - Q)²，大误差时梯度大，易被离群点主导
#    - Huber: |e|≤δ 时等价平方，|e|>δ 时线性，梯度饱和，更稳定
#    - Nature DQN 使用 Huber；delta 大小应与回报尺度匹配
#
# 5. 与 PPO 等对比:
#    ──────────────
#    算法    | 数据使用  | 探索方式        | 目标网络  | 适用动作
#    --------|----------|-----------------|----------|----------
#    DQN     | Off-policy, Replay | ε-greedy | 需要    | 离散
#    PPO     | On-policy | 策略随机性      | 不需要   | 连续/离散
#    SAC     | Off-policy| 熵正则化        | 需要     | 连续
#
# 6. 典型超参数:
#    ───────────
#    gamma = 0.99                # 折扣因子
#    n_step = 1 ~ 10              # n-step return 步数
#    target_update_freq = 1000    # target 更新间隔
#    eps_training = 0.1 ~ 1.0     # 探索率（可随训练衰减）
#    huber_loss_delta = 1.0       # 可选，与回报尺度匹配
#
# 7. 训练技巧:
#    ─────────
#    - 初期可较大 eps 鼓励探索，随训练衰减（如线性或指数）
#    - Replay Buffer 足够大以打破相关性
#    - 若用优先经验回放，weight 由 |td_error| 决定，需在 loss 中乘 weight
#
# ═══════════════════════════════════════════════════════════════════════════════
