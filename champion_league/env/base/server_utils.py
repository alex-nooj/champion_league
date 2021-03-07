from abc import ABC

from server_configuration import ServerConfiguration


class EnvNetwork(ABC):
    """
    Network interface for an environment

    Responsible for communicating with showdown servers. Also implements some higher level
    methods for basic tasks, such as changing avatar and low-level message handling.
    """

    def __init__(
            self,
            *,
            server_configuration: ServerConfiguration,
            start_listening: bool = True,
    ):