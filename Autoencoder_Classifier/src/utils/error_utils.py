class UnequalLenError(Exception):
    """ Custom Error raised when the tokens of the an input do not match the expected length """

    def __init__(self, *args):
        """ constructor """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """ print when raised outside try block """
        message = "The amount of {} passed is not equal to the number of Convolutional Layers."
        if self.message:
            message = message.format(self.message)
        else:
            message = message.format("tokens")

        return message

class InvalidTupleError(Exception):
    """ Custom Error raised when a tuple provided by the user is not of length 2 """

    def __init__(self, *args):
        """ constructor """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        """ print when raised outside try block """
        message = "The tuple {}is not of length 2."
        if self.message:
            message = message.format(self.message + " ")
        else:
            message = message.format("")

        return message
