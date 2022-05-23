# %%
import datetime
import os
import random
import time
from collections import defaultdict
from typing import Callable, Optional

import gym
import numpy as np
import torch
import torch.nn.functional as F
from absl import flags, app
from tensorboardX import SummaryWriter
from torch import nn, optim

import rle_assignment.env
from rle_assignment.utils import LinearSchedule, RingBuffer

# common flags
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('cuda', False, 'Whether to run the model on gpu or on cpu.')
flags.DEFINE_integer('seed', 42, 'Random seed.')

# train flags
flags.DEFINE_float('gamma', .99, 'Discount factor.')
flags.DEFINE_integer('batch_size', 32, 'Train batch size.')
flags.DEFINE_float('learning_rate', 2.5e-4, 'Learning rate.')
flags.DEFINE_float('max_grad_norm', 10, 'Maximum gradient norm. Gradients with larger norms will be clipped.')
flags.DEFINE_integer('num_envs', 1, 'Number of parallel env processes.')
flags.DEFINE_integer('total_steps', 10_000_000, 'Total number of agent steps.')
flags.DEFINE_integer('warmup_steps', 80_000, 'Number of warmup steps to fill the replay buffer.')
flags.DEFINE_integer('buffer_size', 100_000, 'Replay buffer size.')
flags.DEFINE_float('exploration_epsilon_initial', 1.0, 'Initial exploration rate.')
flags.DEFINE_float('exploration_epsilon_final', 0.1, 'Final exploration rate.')
flags.DEFINE_float('exploration_fraction', 0.1, 'Fraction of total_frames it takes to decay initial to final epsilon.')
flags.DEFINE_integer('train_freq', 4, 'Frequency at which train steps are executed.')
flags.DEFINE_integer('checkpoint_freq', 100_000, 'Frequency at which checkpoints are stored.')
flags.DEFINE_integer('logging_freq', 10_000, 'Frequency at which logs are written.')
flags.DEFINE_integer('target_network_update_freq', 1000, 'Frequency at which the target network is updated.')

# eval flags
flags.DEFINE_string('eval_checkpoint', None, 'Eval checkpoint filename.')
flags.DEFINE_integer('eval_num_episodes', 30, 'Number of eval episodes.')
flags.DEFINE_float('eval_epsilon', 0.05, 'Epsilon-greedy during eval.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', 1234, 'Eval seed.')

FLAGS = flags.FLAGS


def make_env_fn(seed: int, render_human: bool = False, video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """ returns a pickleable callable to create an env instance """
    def env_fn():
        env = rle_assignment.env.make_env(render_human, video_folder)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.TransformObservation(env, np.squeeze)  # get rid of 3rd dimension added by ResizeObservation
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return env_fn


class DQN(nn.Module):
    def __init__(self, num_actions: int, in_channels: int = 1):
        super().__init__()
        self.qnet = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, obs):
        if len(obs.shape) == 3:
            obs = torch.unsqueeze(obs, dim=1)  # add channel dim
        obs = obs * (1. / 255.)
        return self.qnet(obs)


def train(device):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    os.makedirs(logdir, exist_ok=False)

    FLAGS.append_flags_into_file(os.path.join(logdir, 'flags.txt'))

    writer = SummaryWriter(os.path.join(logdir, 'logs'))
    writer.add_text("config", FLAGS.flags_into_string())

    envs = gym.vector.SyncVectorEnv([
        make_env_fn(seed=FLAGS.seed, video_folder=os.path.join(logdir, 'videos', 'train') if i == 0 else None)
        for i in range(FLAGS.num_envs)])

    env_name = envs.get_attr('spec')[0].name

    exploration_epsilon_schedule = LinearSchedule(
        initial_value=1.,
        final_value=FLAGS.exploration_epsilon_final,
        schedule_steps=int(FLAGS.exploration_fraction * FLAGS.total_steps)
    )

    q_network = DQN(envs.single_action_space.n).to(device)

    target_network = DQN(envs.single_action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=FLAGS.learning_rate)

    replay_buffer = RingBuffer(size=FLAGS.buffer_size, specs={
        'obs': (envs.single_observation_space.shape, envs.single_observation_space.dtype),
        'next_obs': (envs.single_observation_space.shape, envs.single_observation_space.dtype),
        'actions': (envs.single_action_space.shape, envs.single_action_space.dtype),
        'rewards': ((), np.float32),
        'dones': ((), np.float32),
    })

    logs = defaultdict(list)
    last_log_frame = 0
    last_log_time = time.time()
    total_frames = 0

    obs = envs.reset()

    for global_step in range(FLAGS.total_steps):
        # epsilon greedy action selection
        epsilon = exploration_epsilon_schedule.value(global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        logs["epsilon"].append(epsilon)

        # execute actions in environment
        new_obs, rewards, dones, infos = envs.step(actions)
        for done, info in zip(dones, infos):
            if done and "episode" in info.keys():
                logs[f"{env_name}/episode_frames"].append(info["episode_frame_number"])
                logs[f"{env_name}/episode_reward"].append(info["episode"]["r"])
                logs[f"{env_name}/episode_steps"].append(info["episode"]["l"])
                total_frames += info["episode_frame_number"]

        # vector envs reset automatically, so we have to manually get the terminal observations for these steps
        next_obs = new_obs.copy()
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                next_obs[i] = infos[i]["terminal_observation"]

        # save data to reply buffer
        replay_buffer.put({
            'obs': obs,
            'next_obs': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
        })

        # optimize model (after initial warmup phase to fill the replay buffer)
        if global_step > FLAGS.warmup_steps and global_step % FLAGS.train_freq == 0:
            # sample a batch from the replay buffer
            batch = {
                k: torch.tensor(v).to(device)
                for k, v in replay_buffer.sample(FLAGS.batch_size).items()
            }

            # compute estimate of best q values starting from next states
            next_q_value, _ = target_network(batch['next_obs']).max(dim=1)

            # mask q values where the episode has ended at the current step
            next_q_value_masked = next_q_value * (1 - batch['dones'])

            # compute td target
            td_target = batch['rewards'] + FLAGS.gamma * next_q_value_masked

            # compute estimated q values of actions taken in current step
            selected_q_values = q_network(batch['obs']).gather(1, torch.unsqueeze(batch['actions'], dim=1))

            # compute loss (huber loss)
            loss = F.smooth_l1_loss(selected_q_values.squeeze(), td_target.detach())

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(list(q_network.parameters()), FLAGS.max_grad_norm)
            optimizer.step()

            logs['loss'].append(loss.cpu().detach().numpy())
            logs['q_values'].append(selected_q_values.mean().item())
            logs['grad_norm'].append(grad_norm.cpu().detach().numpy())

        # set obs to new obs for next step
        obs = new_obs

        # update the target network
        if global_step > FLAGS.warmup_steps and global_step % FLAGS.target_network_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        # store checkpoint
        if global_step > FLAGS.warmup_steps and global_step % FLAGS.checkpoint_freq == 0:
            torch.save(q_network.state_dict(), os.path.join(logdir, f'checkpoint-{global_step}.pt'))

        # logging
        if global_step % FLAGS.logging_freq == 0:
            current_log_time = time.time()
            fps = (total_frames - last_log_frame) / (current_log_time - last_log_time)
            writer.add_scalar("fps", fps, total_frames)
            writer.add_scalar('steps', global_step, total_frames)
            for k, v in logs.items():
                writer.add_scalar(f'{k}/mean', np.mean(v), total_frames)
                writer.add_scalar(f'{k}/std', np.std(v), total_frames)
                writer.add_scalar(f'{k}/min', np.min(v), total_frames)
                writer.add_scalar(f'{k}/max', np.max(v), total_frames)
            logs['fps'].append(fps)
            print(" | ".join([f"step={global_step}"] + [f"{k}={np.mean(v):.2f}" for k, v in sorted(logs.items())]))
            logs = defaultdict(list)
            last_log_frame = total_frames
            last_log_time = current_log_time

    envs.close()
    torch.save(q_network.state_dict(), os.path.join(logdir, f'checkpoint-last.pt'))


def eval(device):
    env_fn = make_env_fn(FLAGS.eval_seed, FLAGS.eval_render)
    env = env_fn()

    q_network = DQN(env.action_space.n).to(device)
    q_network.load_state_dict(torch.load(os.path.join(FLAGS.logdir, FLAGS.run_name, FLAGS.eval_checkpoint)))
    q_network.eval()

    episode_rewards = []

    for episode_idx in range(FLAGS.eval_num_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done:
            q_values = q_network(torch.tensor([obs]).to(device))

            if random.random() < FLAGS.eval_epsilon:
                action = env.action_space.sample()
            else:
                action = int(torch.argmax(q_values, dim=1).cpu().numpy())

            obs, reward, done, info = env.step(action)
            step += 1

            if done:
                print(f"Episode {episode_idx}: "
                      f"reward={info['episode']['r']}, "
                      f"steps={step}, "
                      f"frames={info['episode_frame_number']}")
                episode_rewards.append(info['episode']['r'])

    print(f"Evaluation completed: "
          f"mean_episode_reward={np.mean(episode_rewards):.2f}, "
          f"std_episode_reward={np.std(episode_rewards):.2f}, "
          f"min_episode_reward={np.min(episode_rewards):.2f}, "
          f"max_episode_reward={np.max(episode_rewards):.2f}")
    env.close()


def main(_):
    device_name = "cuda" if FLAGS.cuda else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda=true, but cuda is not available")
    device = torch.device(device_name)
    print(f"Using device: {device_name}")

    if FLAGS.mode == 'train':
        train(device)
    elif FLAGS.mode == 'eval':
        eval(device)


if __name__ == "__main__":
    app.run(main)
