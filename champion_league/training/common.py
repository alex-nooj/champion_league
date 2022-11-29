import pathlib
import typing


class MatchMaker:
    def choose_match(self, *args, **kwargs) -> typing.Union[str, pathlib.Path]:
        raise NotImplementedError
