import random

def get_random_easter_egg():
    references_we_have_thought_of_so_far_that_dont_overflow_unsigned_int = [80085, 7734, 836988]
    # sorry Lost numbers :(

    return random.choice(references_we_have_thought_of_so_far_that_dont_overflow_unsigned_int)
