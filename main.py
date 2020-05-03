import PAWS_Bot_Navigation.DQN as DQN
import PAWS_Bot_Navigation.Simulation as sim

def main:
    sim_port = 19999
    sim = Simulation(sim_port)

    abs_state_bot = sim.get_state(sim.paws_bot)

    while (true):
        pass


if __name__ == "__main__":
    main()