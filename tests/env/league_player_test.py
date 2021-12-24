import torch

from champion_league.env import LeaguePlayer
from champion_league.network import AbilityNetwork


class TestLeaguePlayer:
    def test_network_copying(self):
        device = 0 if torch.cuda.is_available() else "cpu"

        data = {
            "x": {
                "2D": torch.zeros((16, 5, 5), device=device),
                "1D": torch.zeros((16, 5), device=device).long(),
            }
        }
        labels = torch.ones((16,), device=device).long()
        loss_fcn = torch.nn.CrossEntropyLoss()
        network = AbilityNetwork(
            nb_actions=2,
            in_shape={"2D": (5, 5), "1D": (5,)},
            embedding_dim=5,
            nb_encoders=1,
            nb_heads=1,
            nb_layers=1,
            scale=False,
            dropout=0.0,
        )
        network = network.to(device)

        optim = torch.optim.Adam(network.parameters(), 0.001)

        lp = LeaguePlayer(
            device=device,
            network=network,
            preprocessor=None,
            sample_moves=True,
        )

        is_equal_tensor = torch.stack(
            [
                torch.all(p == q)
                for p, q in zip(network.parameters(), lp.network.parameters())
            ]
        )

        assert torch.all(is_equal_tensor)

        preds, _ = network(data)
        loss = loss_fcn(preds["rough_action"], labels)
        optim.zero_grad()
        loss.backward()
        optim.step()

        is_equal_tensor = torch.stack(
            [
                torch.all(p == q)
                for p, q in zip(network.parameters(), lp.network.parameters())
            ]
        )

        assert not torch.all(is_equal_tensor)

        lp.update_network(network)

        is_equal_tensor = torch.stack(
            [
                torch.all(p == q)
                for p, q in zip(network.parameters(), lp.network.parameters())
            ]
        )

        assert torch.all(is_equal_tensor)
