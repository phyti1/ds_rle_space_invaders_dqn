from typing import Callable

import gym


def make_env(render_human: bool = False, video_folder: str = None, video_episode_trigger: Callable[[int], bool] = None):
    env = gym.make("ALE/SpaceInvaders-v5",
                   obs_type='grayscale',
                   frameskip=3, 
                   repeat_action_probability=0., full_action_space=False,
                   render_mode='human' if render_human else None)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder, video_episode_trigger)
    return env
