"""
DQN (Deep Q-Network) CartPole训练示例 - 详细注释版

═══════════════════════════════════════════════════════════════════════════════
执行流程Chain概览：
═══════════════════════════════════════════════════════════════════════════════

主要执行阶段：
1. 【初始化阶段】创建环境、网络、策略、算法、Collector
2. 【训练阶段】algorithm.run_training() 启动训练循环
   ├─ Epoch循环 (由 OffPolicyTrainer 管理)
   │  ├─ Training Step (多次)
   │  │  ├─ Collection Step: Collector收集环境数据 → ReplayBuffer
   │  │  └─ Update Step: 从Buffer采样 → 计算loss → 反向传播更新网络
   │  └─ Test Step: 在测试环境评估策略性能
3. 【评估阶段】训练完成后观察训练好的策略

关键数据流：
Environment → Collector → ReplayBuffer → Algorithm.update() → Policy Network
                                                ↓
                                         Target Network (周期性更新)

═══════════════════════════════════════════════════════════════════════════════
"""

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

import tianshou as ts
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import CollectStats
from tianshou.trainer import OffPolicyTrainerParams
from tianshou.utils.net.common import Net
from tianshou.utils.space_info import SpaceInfo


def main() -> None:
    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: 超参数配置
    # ═══════════════════════════════════════════════════════════════════════
    # 这些参数控制整个训练过程的行为
    
    task = "CartPole-v1"  # 环境名称
    
    # --- 优化器参数 ---
    lr = 1e-3  # 学习率：控制每次梯度更新的步长
    
    # --- 训练流程参数 ---
    epoch = 10  # 训练的epoch数量（一个epoch包含多个training step + 一个test step）
    batch_size = 64  # 每次从replay buffer采样的transitions数量（用于一次梯度更新）
    
    # --- 环境数量 ---
    num_training_envs = 10  # 并行训练环境数量（可以加速数据收集）
    num_test_envs = 100  # 并行测试环境数量
    
    # --- DQN核心参数 ---
    gamma = 0.9  # 折扣因子：控制未来奖励的重要性 (0=只看当前, 1=完全考虑未来)
    n_step = 3  # n-step return：用未来n步的实际奖励来估计Q值（减少bias）
    target_freq = 320  # Target网络更新频率：每320次训练更新一次target网络（提供稳定的学习目标）
    
    # --- Buffer参数 ---
    buffer_size = 20000  # Replay Buffer容量：存储最近的20000条transitions
    
    # --- 探索参数 (Epsilon-greedy) ---
    eps_train = 0.1  # 训练时的探索率：10%概率随机选择动作
    eps_test = 0.05  # 测试时的探索率：5%概率随机选择动作（通常更小，更多利用已学知识）
    
    # --- Epoch内的步骤控制 ---
    epoch_num_steps = 10000  # 每个epoch收集的总环境步数
    collection_step_num_env_steps = 10  # 每次collect step收集的环境步数
    
    # ═══════════════════════════════════════════════════════════════════════
    # Step 2: 创建Logger (用于记录训练过程)
    # ═══════════════════════════════════════════════════════════════════════
    logger = ts.utils.TensorboardLogger(SummaryWriter("log/dqn"))
    # TensorBoard会记录：loss、reward、episode length等指标
    # 运行后可用命令查看：tensorboard --logdir log/dqn

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: 创建环境 (Vectorized Environments)
    # ═══════════════════════════════════════════════════════════════════════
    # DummyVectorEnv: 在单进程中顺序执行多个环境（简单但不是真正的并行）
    # SubprocVectorEnv: 每个环境在独立进程中运行（真正的并行，但开销更大）
    
    training_envs = ts.env.DummyVectorEnv(
        [lambda: gym.make(task) for _ in range(num_training_envs)]
    )
    # 用于训练数据收集的环境组
    # 并行收集可以：1) 加速数据收集 2) 增加数据多样性
    
    test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)])
    # 用于策略评估的环境组（评估时运行更多episode以获得稳定的性能估计）

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: 创建神经网络 (Q-Network)
    # ═══════════════════════════════════════════════════════════════════════
    # 获取环境信息（观察空间和动作空间）
    env = gym.make(task, render_mode="human")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    space_info = SpaceInfo.from_env(env)
    state_shape = space_info.observation_info.obs_shape  # CartPole的状态维度 (4,)
    action_shape = space_info.action_info.action_shape  # CartPole的动作数量 (2,) [左/右]
    
    # 创建Q网络：State → [隐藏层] → Q值(每个动作一个)
    net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
    # Net是一个MLP：
    # - 输入：状态 (4维向量)
    # - 隐藏层：3层，每层128个神经元
    # - 输出：每个动作的Q值 (2维向量)
    
    # 创建优化器工厂（用于后续创建优化器实例）
    optim = AdamOptimizerFactory(lr=lr)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: 创建策略 (Policy)
    # ═══════════════════════════════════════════════════════════════════════
    # Policy负责：1) 根据Q网络选择动作 2) 添加探索噪声(epsilon-greedy)
    
    policy = DiscreteQLearningPolicy(
        model=net,  # Q网络
        action_space=env.action_space,
        eps_training=eps_train,  # 训练时探索率
        eps_inference=eps_test,  # 测试/推理时探索率
    )
    # DiscreteQLearningPolicy实现：
    # - forward(): 计算Q值，选择最大Q值对应的动作
    # - add_exploration_noise(): 以eps概率随机选择动作（epsilon-greedy）
    
    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: 创建DQN算法对象
    # ═══════════════════════════════════════════════════════════════════════
    # Algorithm封装了完整的训练逻辑：数据预处理、loss计算、网络更新
    
    algorithm = ts.algorithm.DQN(
        policy=policy,
        optim=optim,
        gamma=gamma,  # 折扣因子
        n_step_return_horizon=n_step,  # n-step TD target
        target_update_freq=target_freq,  # Target网络更新频率
    )
    # DQN算法的关键机制：
    # 1. Experience Replay: 从buffer中随机采样，打破数据相关性
    # 2. Target Network: 使用独立的target网络计算TD target，每target_freq步更新一次
    # 3. n-step Return: 使用n步的实际奖励，减少bootstrapping的bias
    
    # ═══════════════════════════════════════════════════════════════════════
    # Step 7: 创建数据收集器 (Collector)
    # ═══════════════════════════════════════════════════════════════════════
    # Collector负责：与环境交互 → 收集transitions → 存入buffer
    
    training_collector = ts.data.Collector[CollectStats](
        algorithm,  # 使用algorithm中的policy来选择动作
        training_envs,  # 训练环境
        ts.data.VectorReplayBuffer(buffer_size, num_training_envs),  # Replay Buffer
        exploration_noise=True,  # 启用exploration noise (epsilon-greedy)
    )
    # VectorReplayBuffer: 为每个并行环境维护独立的buffer
    # exploration_noise=True: 收集数据时使用epsilon-greedy探索
    
    test_collector = ts.data.Collector[CollectStats](
        algorithm,
        test_envs,
        exploration_noise=True,  # DQN测试时也需要一定探索（但eps更小）
    )
    # 测试collector不需要buffer，因为只是评估性能，不需要存储数据
    
    # ═══════════════════════════════════════════════════════════════════════
    # Step 8: 定义早停函数 (Early Stopping)
    # ═══════════════════════════════════════════════════════════════════════
    def stop_fn(mean_rewards: float) -> bool:
        """
        判断是否达到训练目标，可以提前停止训练
        """
        if env.spec:
            if not env.spec.reward_threshold:
                return False
            else:
                # 如果平均reward达到环境定义的阈值，停止训练
                return mean_rewards >= env.spec.reward_threshold
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Step 9: 启动训练 (Main Training Loop)
    # ═══════════════════════════════════════════════════════════════════════
    # 这是整个训练的核心！algorithm.run_training()会执行：
    #
    # 训练循环结构：
    # for epoch in range(max_epochs):
    #     while epoch_steps < epoch_num_steps:
    #         # === Collection Step ===
    #         # Collector与环境交互收集数据
    #         for step in range(collection_step_num_env_steps):
    #             obs → policy.forward() → action
    #             action → env.step() → (next_obs, reward, done, info)
    #             store (obs, action, reward, next_obs, done) into buffer
    #         
    #         # === Update Step ===
    #         # 从buffer采样，计算loss，更新网络
    #         num_gradient_steps = collection_step_num_env_steps * update_step_num_gradient_steps_per_sample
    #         for _ in range(num_gradient_steps):
    #             batch = buffer.sample(batch_size)  # 采样64条transitions
    #             
    #             # 计算当前Q值
    #             q_current = policy(batch.obs).logits[batch.act]
    #             
    #             # 计算target Q值（使用target network）
    #             q_next = target_network(batch.obs_next).max()
    #             q_target = batch.reward + gamma * q_next * (1 - batch.done)
    #             
    #             # 计算TD error和loss
    #             td_error = q_target - q_current
    #             loss = td_error^2
    #             
    #             # 反向传播，更新Q网络
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             
    #             # 周期性更新target network
    #             if step % target_freq == 0:
    #                 target_network.load_state_dict(q_network.state_dict())
    #     
    #     # === Test Step ===
    #     # 在测试环境评估当前策略
    #     test_collector.collect(n_episode=num_test_envs)
    #     log test performance
    
    result = algorithm.run_training(
        OffPolicyTrainerParams(
            # --- Collectors ---
            training_collector=training_collector,  # 用于收集训练数据
            test_collector=test_collector,  # 用于评估策略
            
            # --- Epoch设置 ---
            max_epochs=epoch,  # 总共训练10个epoch
            epoch_num_steps=epoch_num_steps,  # 每个epoch收集10000步
            
            # --- Collection Step设置 ---
            collection_step_num_env_steps=collection_step_num_env_steps,  # 每次collect 10步
            # 这意味着每个epoch会有 10000/10 = 1000个training steps
            
            # --- Test设置 ---
            test_step_num_episodes=num_test_envs,  # 每次测试运行100个episode
            
            # --- Update Step设置 ---
            batch_size=batch_size,  # 每次梯度更新使用64条transitions
            update_step_num_gradient_steps_per_sample=1 / collection_step_num_env_steps,
            # 每收集1个样本，执行 1/10 = 0.1 次梯度更新
            # 即：每collect 10步，执行 10 * 0.1 = 1次梯度更新
            
            # --- 回调函数 ---
            stop_fn=stop_fn,  # 早停函数
            logger=logger,  # 日志记录器
            test_in_training=True,  # 在训练过程中也进行测试（用于early stopping）
        )
    )
    # run_training()返回的result包含：
    # - timing: 训练耗时统计
    # - best_score: 最佳测试分数
    # - best_reward: 最佳测试reward
    
    print(f"Finished training in {result.timing.total_time} seconds")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 10: 观察训练好的策略 (Watch Performance)
    # ═══════════════════════════════════════════════════════════════════════
    # 训练完成后，让agent在环境中运行并可视化
    
    collector = ts.data.Collector[CollectStats](algorithm, env, exploration_noise=True)
    # 创建新的collector用于可视化（使用render_mode="human"的环境）
    
    collector.collect(n_episode=100, render=1 / 35, reset_before_collect=True)
    # n_episode=100: 运行100个episode
    # render=1/35: 渲染速度（约35 FPS）
    # reset_before_collect=True: 收集前重置环境
    
    # 执行流程：
    # for episode in range(100):
    #     obs = env.reset()
    #     while not done:
    #         action = policy(obs)  # 使用训练好的策略选择动作
    #         obs, reward, done, info = env.step(action)
    #         env.render()  # 渲染可视化


if __name__ == "__main__":
    main()

# ═══════════════════════════════════════════════════════════════════════════════
# 关键概念总结
# ═══════════════════════════════════════════════════════════════════════════════
#
# 1. Q-Learning核心思想：
#    学习一个Q函数 Q(s,a)，表示在状态s采取动作a的长期价值
#    最优策略：π*(s) = argmax_a Q*(s,a)
#
# 2. DQN的三大创新：
#    a) Experience Replay: 打破数据相关性，提高样本效率
#    b) Target Network: 提供稳定的学习目标，减少震荡
#    c) 使用深度神经网络近似Q函数，可处理高维状态
#
# 3. TD Learning (Temporal Difference):
#    Q(s,a) ← Q(s,a) + α[r + γ*max_a'Q(s',a') - Q(s,a)]
#    其中 r + γ*max_a'Q(s',a') 是TD target
#
# 4. n-step Return:
#    使用n步的实际奖励代替单步：
#    G_t^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*max_a Q(s_{t+n}, a)
#    好处：减少bias（更接近真实return），但会增加variance
#
# 5. Epsilon-Greedy探索：
#    以概率ε随机选择动作（探索）
#    以概率1-ε选择Q值最大的动作（利用）
#    训练时ε较大（如0.1），测试时ε较小（如0.05）
#
# 6. Off-Policy学习：
#    收集数据的策略（behavior policy）和优化的策略（target policy）可以不同
#    DQN使用epsilon-greedy收集数据，但优化的是greedy policy
#
# 7. Replay Buffer作用：
#    - 存储历史transitions，可重复使用（提高样本效率）
#    - 随机采样打破时间相关性（提高训练稳定性）
#    - 允许off-policy学习
#
# 8. 训练流程数据维度示例（以CartPole为例）：
#    - State: (4,) [位置, 速度, 角度, 角速度]
#    - Action: 离散{0, 1} [左, 右]
#    - Q值: (2,) [Q(s,左), Q(s,右)]
#    - Batch: (64, 4) 状态 + (64,) 动作 + (64,) 奖励 + (64, 4) 下一状态
#
# ═══════════════════════════════════════════════════════════════════════════════
