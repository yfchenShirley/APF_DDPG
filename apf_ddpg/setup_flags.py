import argparse

# Function takes input from user to determine mode of agent.  Options include whether agent will be in training (noise added to policy), testing (no noise added), or a mix.  User can also indicate whether episode should be visualized.

def set_up():
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--exp',
        default='exp1',
        type=str,
        help='Experiment names'
    )


    parser.add_argument(
        '--evaluate',
        default=False,
        action='store_true',
        help='Evaluate the trained agent?'
    )

    parser.add_argument(
        '--show',
        default=False,
        action='store_true',
        help='Render the simulation environment?'
    )


    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS
