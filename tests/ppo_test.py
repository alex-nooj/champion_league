import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

from champion_league.agent.ppo import PPOAgent
from champion_league.network import build_network_from_args
from champion_league.network import NETWORKS
from champion_league.utils.directory_utils import DotDict
from champion_league.utils.replay import Episode
from champion_league.utils.replay import History

resize = T.Compose(
    [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
)
REWARD_SCALE = 100


def test_networks(network_type: str, env: str, expected_return: int):
    env = gym.make(env).unwrapped

    def get_cart_location(screen_width: int) -> int:
        world_width = env.x_threshold * 2
        scale = screen_width / world_width
        return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen() -> torch.Tensor:
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode="rgb_array").transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4) : int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(
                cart_location - view_width // 2, cart_location + view_width // 2
            )
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = (
            0.2989 * screen[0, :, :]
            + 0.5870 * screen[1, :, :]
            + 0.1140 * screen[2, :, :]
        )
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nb_actions = env.action_space.n
    env.reset()

    init_screen = get_screen()
    _, screen_height, screen_width = init_screen.shape

    in_shape = {"2D": (screen_height, screen_width), "1D": (1,)}

    network = build_network_from_args(
        DotDict(
            {
                "nb_actions": nb_actions,
                "in_shape": in_shape,
                "device": device,
                "network": network_type,
                "embedding_dim": 1,
            }
        )
    ).eval()

    agent = PPOAgent(
        device=device,
        network=network,
        lr=0.007,
        entropy_weight=0.01,
        clip=0.2,
        logdir="/tmp/",
        tag=f"test_{network_type}",
    )
    history = History()
    done = False
    internals = agent.network.reset(device)
    episode = Episode()
    episode_rewards = [0] * 100
    current_screen = get_screen()
    # last_screen = get_screen()

    observation = {
        # "2D": current_screen - last_screen,
        "2D": current_screen,
        "1D": torch.zeros((1, screen_height), device=device, dtype=torch.int),
    }
    for i in range(5_000_000):
        if done:
            total_return = REWARD_SCALE * float(np.sum(episode.rewards))
            episode_rewards[(i % 100)] = total_return
            if float(np.mean(episode_rewards)) >= expected_return:
                break
            print(
                f"Step {i:6d}: {total_return:3.1f} Return, {float(np.mean(episode_rewards)):3.1f} Average"
            )
            agent.write_to_tboard("Reward", total_return)
            agent.write_to_tboard("Average Return", float(np.mean(episode_rewards)))

            episode.end_episode(last_value=0)
            history.add_episode(episode)

            if len(history) > (64 * 20):
                history.build_dataset()
                data_loader = DataLoader(history, 64, shuffle=True, drop_last=True)
                epoch_losses = agent.learn_step(data_loader)
                for key, v in epoch_losses.items():
                    for val in v:
                        agent.write_to_tboard(f"Loss/{key}", val)
                history.free_memory()

            internals = agent.network.reset(device=device)
            env.reset()
            # last_screen = get_screen()
            current_screen = get_screen()
            observation = {
                # "2D": current_screen - last_screen,
                "2D": current_screen,
                "1D": torch.zeros((1, screen_height), device=device, dtype=torch.int),
            }
            episode = Episode()

        action, log_prob, value, new_internals = agent.sample_action(
            observation, internals
        )

        _, reward, done, _ = env.step(action)

        current_screen = get_screen()
        new_observation = {
            # "2D": current_screen - last_screen,
            "2D": current_screen,
            "1D": torch.zeros((1, screen_height), device=device, dtype=torch.int),
        }

        episode.append(
            observation=observation,
            internals=internals,
            action=action,
            reward=reward,
            value=value,
            log_probability=log_prob,
            reward_scale=REWARD_SCALE,
        )

        observation = new_observation
        internals = new_internals

    assert float(np.mean(episode_rewards)) >= expected_return


if __name__ == "__main__":
    for network_name in NETWORKS:
        if network_name != "EncoderLSTM":
            continue
        try:
            test_networks(
                network_type=network_name, env="CartPole-v0", expected_return=195
            )
        except AssertionError:
            print(f"{network_name} did not reach the expected return.")
