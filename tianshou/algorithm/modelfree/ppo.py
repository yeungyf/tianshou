"""
PPO (Proximal Policy Optimization) 算法实现 - 详细注释版

═══════════════════════════════════════════════════════════════════════════════
算法概览
═══════════════════════════════════════════════════════════════════════════════

PPO是一种On-Policy的策略梯度算法，解决了TRPO（Trust Region Policy Optimization）
计算复杂度高的问题，同时保持了稳定的训练过程。

核心思想：
- 限制策略更新的幅度，防止性能崩溃
- 使用简单的clipping机制代替TRPO的KL散度约束
- 多次使用同一批数据进行更新（通过importance sampling）

═══════════════════════════════════════════════════════════════════════════════
PPO vs DQN 对比
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

训练一个epoch的流程：
1. 【数据收集】Collector用当前策略收集N步数据 → Buffer
2. 【预处理】_preprocess_batch:
   - 计算GAE优势函数 A(s,a)
   - 保存old policy的log π_old(a|s)
3. 【多轮更新】_update_with_batch (重复K次):
   for each mini-batch:
     a) 用新策略计算 log π_new(a|s)
     b) 计算importance ratio: r = π_new / π_old
     c) 计算clipped objective: L^CLIP = min(r*A, clip(r)*A)
     d) 计算value loss: L^VF = (V(s) - return)²
     e) 计算entropy bonus: L^ENT = H(π)
     f) 总loss = L^CLIP + c1*L^VF - c2*L^ENT
     g) 反向传播，更新网络
4. 【丢弃数据】清空buffer，下一个epoch重新收集

关键数据流：
Environment → Collector → Buffer → PPO.preprocess (计算GAE & logp_old)
                                      ↓
                            PPO.update (多轮mini-batch更新)
                                      ↓
                            Policy & Critic 网络更新

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import cast

import numpy as np
import torch

from tianshou.algorithm import A2C
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
from tianshou.utils.net.continuous import ContinuousCritic
from tianshou.utils.net.discrete import DiscreteCritic


class PPO(A2C):
    """
    PPO (Proximal Policy Optimization) 算法实现
    
    论文: https://arxiv.org/abs/1707.06347
    
    PPO的核心创新是Clipped Surrogate Objective:
    L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
    
    其中:
    - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (importance sampling ratio)
    - Â_t 是优势函数的估计值
    - ε 是clipping参数（通常0.2）
    - clip(x, min, max) 将x限制在[min, max]范围内
    
    这个目标函数确保新策略不会偏离旧策略太远，防止训练不稳定。
    """

    def __init__(
        self,
        *,
        policy: ProbabilisticActorPolicy,
        critic: torch.nn.Module | ContinuousCritic | DiscreteCritic,
        optim: OptimizerFactory,
        eps_clip: float = 0.2,
        dual_clip: float | None = None,
        value_clip: bool = False,
        advantage_normalization: bool = True,
        recompute_advantage: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        gae_lambda: float = 0.95,
        max_batchsize: int = 256,
        gamma: float = 0.99,
        return_scaling: bool = False,
    ) -> None:
        r"""
        PPO算法初始化
        
        ═══════════════════════════════════════════════════════════════════════
        关键参数说明
        ═══════════════════════════════════════════════════════════════════════
        
        :param policy: 策略网络（Actor），输出动作的概率分布
        :param critic: 价值网络（Critic），输出状态价值 V(s)
        :param optim: 优化器工厂，用于创建优化器
        
        --- PPO特有参数 ---
        :param eps_clip: Clipping参数ε，控制策略更新幅度
            - 典型值: 0.1 ~ 0.3，论文推荐0.2
            - 作用: ratio被clip到 [1-ε, 1+ε]
            - 越小越保守，更新越稳定但可能慢
            
        :param dual_clip: 双重clipping参数（可选）
            - 防止对负优势的动作过度悲观
            - 当A<0时: L = max(min(r*A, clip(r)*A), c*A)
            - 典型值: 2.0 ~ 5.0，或设为None禁用
            - 适用场景: 需要更多探索的环境
            
        :param value_clip: 是否对价值函数也进行clipping
            - True: V_clip = V_old + clip(V_new - V_old, -ε, ε)
            - 防止价值函数剧烈变化
            - 提高稳定性但可能减慢收敛
            
        :param advantage_normalization: 是否归一化优势函数
            - True: A' = (A - mean(A)) / std(A)
            - 让不同batch的优势值在同一尺度
            - 通常建议开启
            
        :param recompute_advantage: 是否在每次重复更新时重新计算优势
            - True: 每轮都用最新的V(s)计算GAE
            - 可能提高性能但增加计算量
            - 参考: https://arxiv.org/pdf/2006.05990.pdf
            
        --- A2C继承参数 ---
        :param vf_coef: Value loss的系数（c1），通常0.5 ~ 1.0
        :param ent_coef: Entropy bonus的系数（c2），通常0.01 ~ 0.05
        :param max_grad_norm: 梯度裁剪的最大L2范数，防止梯度爆炸
        :param gae_lambda: GAE的λ参数，控制bias-variance权衡
            - 0: 纯TD误差（高bias，低variance）
            - 1: 纯MC return（低bias，高variance）
            - 典型值: 0.95 ~ 0.99
        :param max_batchsize: 计算GAE时的最大batch大小
        :param gamma: 折扣因子γ，通常0.99
        :param return_scaling: 是否对return进行标准化
        
        ═══════════════════════════════════════════════════════════════════════
        总Loss公式
        ═══════════════════════════════════════════════════════════════════════
        
        L_total = L^CLIP + c1 * L^VF - c2 * L^ENT
        
        其中:
        - L^CLIP: Clipped policy loss (要最大化，所以取负)
        - L^VF: Value function loss (MSE)
        - L^ENT: Entropy (要最大化，所以取负)
        
        ═══════════════════════════════════════════════════════════════════════
        """
        # 验证dual_clip参数的有效性
        assert dual_clip is None or dual_clip > 1.0, (
            f"Dual-clip PPO parameter should greater than 1.0 but got {dual_clip}"
        )

        # 调用父类A2C的初始化
        # PPO继承了A2C的GAE计算、价值函数训练等功能
        super().__init__(
            policy=policy,
            critic=critic,
            optim=optim,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            max_batchsize=max_batchsize,
            gamma=gamma,
            return_scaling=return_scaling,
        )
        
        # PPO特有的参数
        self.eps_clip = eps_clip  # Clipping范围: [1-ε, 1+ε]
        self.dual_clip = dual_clip  # 双重clipping参数（可选）
        self.value_clip = value_clip  # 是否对价值函数clipping
        self.advantage_normalization = advantage_normalization  # 优势归一化
        self.recompute_adv = recompute_advantage  # 重新计算优势

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        """
        预处理batch数据，为PPO训练做准备
        
        ═══════════════════════════════════════════════════════════════════════
        执行步骤 (Preprocessing Chain)
        ═══════════════════════════════════════════════════════════════════════
        
        Step 1: 保存buffer和indices（如果需要重新计算优势）
        Step 2: 计算returns和advantages（调用父类A2C的方法）
                - 使用GAE (Generalized Advantage Estimation)
                - GAE(λ): A_t = Σ(γλ)^k δ_{t+k}
                - 其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
        Step 3: 转换动作为torch tensor
        Step 4: 计算old policy的log probabilities
                - 使用旧策略（训练前的策略）
                - 分batch计算以节省内存
                - 保存 logp_old = log π_old(a|s)
        
        ═══════════════════════════════════════════════════════════════════════
        为什么需要logp_old？
        ═══════════════════════════════════════════════════════════════════════
        
        PPO使用importance sampling在多次更新中重用数据：
        - 数据是用旧策略π_old收集的
        - 但我们要更新新策略π_new
        - ratio = π_new(a|s) / π_old(a|s) 用于纠正分布不匹配
        - log space更稳定: ratio = exp(log π_new - log π_old)
        
        ═══════════════════════════════════════════════════════════════════════
        
        :param batch: 收集到的rollout数据
        :param buffer: Replay buffer（虽然是on-policy，但仍用buffer暂存）
        :param indices: Buffer中的数据索引
        :return: 增加了logp_old字段的batch
        """
        # Step 1: 如果需要重新计算优势，保存buffer和indices
        if self.recompute_adv:
            # 保存这些，以便在_update_with_batch中使用
            self._buffer, self._indices = buffer, indices
        
        # Step 2: 计算returns（回报）和advantages（优势函数）
        # 这是从父类A2C继承的方法，使用GAE算法
        # 计算结果：
        # - batch.returns: 估计的回报 G_t
        # - batch.adv: 优势函数 A(s,a) = Q(s,a) - V(s)
        # - batch.v_s: 状态价值 V(s)
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        
        # Step 3: 转换动作为torch tensor（确保类型匹配）
        batch.act = to_torch_as(batch.act, batch.v_s)
        
        # Step 4: 计算old policy的log probabilities
        # 这是PPO的关键：保存数据收集时策略的log概率
        logp_old = []
        with torch.no_grad():  # 不需要梯度，只是记录旧策略的行为
            # 分成mini-batch处理，避免内存溢出
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                # self.policy(minibatch).dist: 当前策略对minibatch的动作分布
                # .log_prob(minibatch.act): 这些动作在分布下的log概率
                # 注意：虽然调用的是当前policy，但此时policy还没开始更新
                # 所以这实际上是"收集数据时的策略"，即old policy
                logp_old.append(self.policy(minibatch).dist.log_prob(minibatch.act))
            
            # 合并所有mini-batch的结果，flatten为1D tensor
            batch.logp_old = torch.cat(logp_old, dim=0).flatten()
        
        # 返回增强后的batch，包含：obs, act, rew, returns, adv, v_s, logp_old
        return cast(LogpOldProtocol, batch)

    def _update_with_batch(  # type: ignore[override]
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        """
        PPO的核心训练逻辑 - 使用Clipped Surrogate Objective
        
        ═══════════════════════════════════════════════════════════════════════
        训练流程 (Update Chain)
        ═══════════════════════════════════════════════════════════════════════
        
        外层循环: 重复K次（通常3-10次）
          内层循环: 遍历所有mini-batches
            Step 1: [可选] 重新计算优势函数
            Step 2: 计算importance sampling ratio: r = π_new / π_old
            Step 3: [可选] 归一化优势函数
            Step 4: 计算PPO的clipped objective loss
            Step 5: 计算value function loss
            Step 6: 计算entropy bonus
            Step 7: 组合总loss并更新网络参数
        
        ═══════════════════════════════════════════════════════════════════════
        为什么要重复K次？
        ═══════════════════════════════════════════════════════════════════════
        
        - On-policy算法通常样本效率低（每次只用新数据）
        - PPO通过importance sampling允许重用数据
        - Clipping机制防止更新太大，保证安全
        - 实践中repeat=3~10通常效果最好
        - 太多次会导致分布偏移过大
        
        ═══════════════════════════════════════════════════════════════════════
        
        :param batch: 预处理后的batch（包含logp_old）
        :param batch_size: Mini-batch大小（用于SGD）
        :param repeat: 重复更新次数（epoch数）
        :return: 训练统计信息
        """
        # 用于记录各种loss
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0  # 梯度更新总次数
        split_batch_size = batch_size or -1  # -1表示不分割
        
        # ═══════════════════════════════════════════════════════════════════
        # 外层循环：多轮训练（重复使用同一批数据）
        # ═══════════════════════════════════════════════════════════════════
        for step in range(repeat):
            # Step 1: [可选] 重新计算优势函数
            # 如果启用recompute_advantage，在每轮都用最新的critic网络重新计算GAE
            # 这样可以使用更准确的价值估计，但会增加计算开销
            if self.recompute_adv and step > 0:
                batch = cast(
                    LogpOldProtocol,
                    self._add_returns_and_advantages(batch, self._buffer, self._indices),
                )
            
            # ═══════════════════════════════════════════════════════════════
            # 内层循环：遍历mini-batches（Mini-batch SGD）
            # ═══════════════════════════════════════════════════════════════
            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                
                # ═══════════════════════════════════════════════════════════
                # Step 2 & 3: 计算importance sampling ratio和归一化优势
                # ═══════════════════════════════════════════════════════════
                
                # 获取优势函数 A(s,a)
                advantages = minibatch.adv
                
                # 用当前（新）策略计算动作的分布
                dist = self.policy(minibatch).dist
                
                # Step 3: [可选] Per mini-batch归一化优势
                # 归一化后优势值在同一尺度，有助于稳定训练
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)  # 避免除零
                
                # Step 2: 计算importance sampling ratio
                # ratio = π_new(a|s) / π_old(a|s)
                # 在log space: ratio = exp(log π_new - log π_old)
                ratios = (dist.log_prob(minibatch.act) - minibatch.logp_old).exp().float()
                # Reshape以匹配advantages的维度
                ratios = ratios.reshape(ratios.size(0), -1).transpose(0, 1)
                
                # ═══════════════════════════════════════════════════════════
                # Step 4: 计算PPO的Clipped Surrogate Objective
                # ═══════════════════════════════════════════════════════════
                
                # PPO的核心思想：限制ratio在[1-ε, 1+ε]范围内
                # surr1: 原始的policy gradient目标
                # surr2: Clipped版本
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
                
                # 根据是否启用dual_clip选择不同的objective
                if self.dual_clip:
                    # ═══════════════════════════════════════════════════════
                    # Dual Clipping（双重裁剪）
                    # ═══════════════════════════════════════════════════════
                    # 目的：防止对负优势的动作过度悲观
                    # 
                    # 当A<0时（坏动作）：
                    #   - 单clip: min(r*A, clip(r)*A) 可能过度惩罚
                    #   - 双clip: max(min(r*A, clip(r)*A), c*A) 设置惩罚下限
                    # 
                    # 效果：保留一定的探索能力
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    # 根据优势的符号选择使用哪个clip
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    # ═══════════════════════════════════════════════════════
                    # 标准PPO Clipping（单clip）
                    # ═══════════════════════════════════════════════════════
                    # L^CLIP = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
                    # 
                    # 直观理解：
                    # - 当A>0（好动作）且r>1+ε：clip到1+ε，限制奖励增加
                    # - 当A>0（好动作）且r<1-ε：clip到1-ε，限制奖励减少
                    # - 当A<0（坏动作）：类似逻辑，限制惩罚变化
                    # 
                    # 效果：防止策略变化太大，保证训练稳定
                    clip_loss = -torch.min(surr1, surr2).mean()
                
                # ═══════════════════════════════════════════════════════════
                # Step 5: 计算Value Function Loss
                # ═══════════════════════════════════════════════════════════
                
                # 用critic网络预测状态价值
                value = self.critic(minibatch.obs).flatten()
                
                if self.value_clip:
                    # ═══════════════════════════════════════════════════════
                    # Value Clipping（价值裁剪）
                    # ═══════════════════════════════════════════════════════
                    # 限制价值函数的变化幅度
                    # v_clip = v_old + clip(v_new - v_old, -ε, ε)
                    # 
                    # 取max是悲观估计：
                    # L^VF = max((return - v)², (return - v_clip)²)
                    # 即：如果clipping让误差更大，就用clipping的误差
                    # 
                    # 效果：防止价值函数剧烈变化
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)  # 未clip的MSE
                    vf2 = (minibatch.returns - v_clip).pow(2)  # clipped的MSE
                    vf_loss = torch.max(vf1, vf2).mean()  # 悲观选择
                else:
                    # ═══════════════════════════════════════════════════════
                    # 标准MSE Loss（无clipping）
                    # ═══════════════════════════════════════════════════════
                    # L^VF = (V(s) - return)²
                    # 标准的价值函数回归损失
                    vf_loss = (minibatch.returns - value).pow(2).mean()
                
                # ═══════════════════════════════════════════════════════════
                # Step 6: 计算Entropy Bonus
                # ═══════════════════════════════════════════════════════════
                # 熵衡量策略的随机性
                # - 高熵：动作分布均匀，更多探索
                # - 低熵：动作分布集中，更多利用
                # 
                # 在loss中减去熵（即最大化熵），鼓励探索
                ent_loss = dist.entropy().mean()
                
                # ═══════════════════════════════════════════════════════════
                # Step 7: 组合总Loss并更新
                # ═══════════════════════════════════════════════════════════
                # L_total = L^CLIP + c1*L^VF - c2*L^ENT
                # 
                # 其中：
                # - L^CLIP: 策略loss（已取负，所以是最小化 -reward）
                # - c1*L^VF: 价值函数loss（MSE，最小化预测误差）
                # - c2*L^ENT: 熵bonus（取负，最大化熵）
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                
                # 执行一步梯度更新
                # self.optim.step()会：
                # 1. 计算梯度：loss.backward()
                # 2. [可选] 梯度裁剪：防止梯度爆炸
                # 3. 更新参数：optimizer.step()
                # 4. 清零梯度：optimizer.zero_grad()
                self.optim.step(loss)
                
                # 记录各项loss（用于监控训练）
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        # ═══════════════════════════════════════════════════════════════════
        # 返回训练统计信息
        # ═══════════════════════════════════════════════════════════════════
        return A2CTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),  # 总loss统计
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),  # 策略loss
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),  # 价值loss
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),  # 熵loss
            gradient_steps=gradient_steps,  # 总梯度步数
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PPO算法关键概念总结
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Clipped Surrogate Objective（核心创新）:
#    ────────────────────────────────────────
#    L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
#    
#    其中 r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
#    
#    直观理解：
#    - ratio > 1: 新策略增加了这个动作的概率
#    - ratio < 1: 新策略减少了这个动作的概率
#    - clipping: 限制ratio的变化范围，防止更新太激进
#    
#    为什么有效？
#    - 保守更新：不会一步跳太远
#    - 简单高效：只需要简单的clip操作
#    - 效果好：接近TRPO但计算量小得多
#
# 2. Generalized Advantage Estimation (GAE):
#    ─────────────────────────────────────────
#    A^GAE(γ,λ)_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
#    
#    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
#    
#    λ的作用：
#    - λ=0: A_t = δ_t (1-step TD, 高bias低variance)
#    - λ=1: A_t = Σ γ^l δ_{t+l} (MC, 低bias高variance)
#    - λ∈(0,1): 平衡bias和variance
#
# 3. Importance Sampling（重要性采样）:
#    ──────────────────────────────────
#    允许使用旧策略收集的数据训练新策略
#    
#    期望修正：E_π_old[r(θ)·f(x)] = E_π_new[f(x)]
#    
#    但要注意：
#    - ratio太大会导致高方差
#    - 所以PPO用clipping限制ratio
#
# 4. 与其他算法对比:
#    ────────────────
#    
#    算法    | 数据使用  | 更新策略      | 稳定性  | 计算复杂度
#    --------|----------|--------------|---------|----------
#    REINFORCE| On-policy| 无约束       | 差      | 低
#    A2C     | On-policy| 无约束       | 中      | 低
#    TRPO    | On-policy| KL散度约束   | 好      | 高
#    PPO     | On-policy| Clipping约束 | 好      | 中
#    DQN     | Off-policy| -           | 中      | 低
#    SAC     | Off-policy| 熵正则化     | 好      | 中
#
# 5. PPO的优势:
#    ──────────
#    ✓ 简单：只需要clip操作，易于实现
#    ✓ 稳定：保守更新策略，训练平滑
#    ✓ 高效：可重用数据，减少环境交互
#    ✓ 通用：连续/离散动作空间都适用
#    ✓ 少调参：对超参数不太敏感
#
# 6. 典型超参数设置:
#    ────────────────
#    eps_clip = 0.2              # Clipping范围
#    vf_coef = 0.5               # Value loss系数
#    ent_coef = 0.01             # Entropy系数
#    gae_lambda = 0.95           # GAE的λ
#    gamma = 0.99                # 折扣因子
#    repeat = 4                  # 重复更新次数
#    batch_size = 64             # Mini-batch大小
#    max_grad_norm = 0.5         # 梯度裁剪
#    advantage_normalization = True  # 优势归一化
#
# 7. 训练技巧:
#    ─────────
#    • 开始时可以用较大的ent_coef鼓励探索
#    • 如果训练不稳定，减小eps_clip
#    • 复杂环境可能需要更多的repeat
#    • Advantage normalization通常有帮助
#    • 可以尝试学习率调度（cosine annealing等）
#
# ═══════════════════════════════════════════════════════════════════════════════
