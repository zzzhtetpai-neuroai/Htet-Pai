from random import randint

def roll_dice(sides=6):
    """Simulate rolling a dice with a given number of sides."""
    return randint(1, sides)