
class IncorrectBasePathDir(Exception):
    """Exception raised for errors in the base path.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="The base dir in config file is incorrect"):
        self.message = message
        super().__init__(self.message)
