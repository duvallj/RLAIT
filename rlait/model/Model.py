import keras

class Model:
    """
    A helper class to easily add Neural Networks to
    approaches. Just another interface to facilitate changing
    out different models easily.
    """

    def __init__(self, args, input_layer):
        """
        Being passed in an input layer, then stored in `self.input`,
        defines a single keras output layer, stored in `self.output`.

        These can be used to construct the rest of the model in conjuction with
        the task. These classes can be seen as keras helper classes.
        """
        pass
