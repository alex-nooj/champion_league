from champion_league.utils.replay import Episode


class ReplayBuffer:
    def add_episode(self, episode: Episode):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
