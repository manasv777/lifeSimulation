import mesa
import numpy as np
import matplotlib.pyplot as plt
import random as pyrandom


class WormAgent(mesa.Agent):

    def __init__(self, model):
        super().__init__(model)
        self.hunger = 0.0

    def step(self):
        self.move()
        self.hunger += 0.01

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=False,
            include_center=True
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)


class WormWorld(mesa.Model):

    def __init__(self, width=20, height=20, seed = None):
        super().__init__()
        self.rng = pyrandom.Random(seed)

        self.width = width
        self.height = height

        self.grid = mesa.space.SingleGrid(width, height, torus=False)

        # Place worm at random position
        worm_x = self.rng.randint(0, self.width - 1)
        worm_y = self.rng.randint(0, self.height - 1)
        worm_pos = (worm_x, worm_y)

        # Place food at random position, not on worm
        while True:
            food_x = self.rng.randint(1, self.width - 2)
            food_y = self.rng.randint(1, self.height - 2)
            if (food_x, food_y) != worm_pos:
                self.food_pos = (food_x, food_y)
                break

        # Place noxious zone at random position, not on worm or food
        while True:
            nox_x = self.rng.randint(1, self.width - 2)
            nox_y = self.rng.randint(1, self.height - 2)
            if (nox_x, nox_y) != worm_pos and (nox_x, nox_y) != self.food_pos:
                self.noxious_pos = (nox_x, nox_y)
                break

        self.noxious_gradient = np.zeros((width, height))
        self._update_noxious_gradient()

        self.gradient = np.zeros((width, height))
        self._update_gradient()

        worm = WormAgent(self)
        self.grid.place_agent(worm, worm_pos)

    def _update_gradient(self):
        for x in range(self.width):
            for y in range(self.height):
                dist = np.sqrt(
                    (x - self.food_pos[0])**2 +
                    (y - self.food_pos[1])**2
                )
                self.gradient[x][y] = 1.0 / (1.0 + dist)

    def _update_noxious_gradient(self):
        for x in range(self.width):
            for y in range(self.height):
                dist = np.sqrt(
                    (x - self.noxious_pos[0])**2 +
                    (y - self.noxious_pos[1])**2
                )
                self.noxious_gradient[x][y] = 1.0 / (1.0 + dist * 0.3)

    def step(self):
        self.agents.shuffle_do("step")


def visualize_world(world):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(world.gradient.T, origin='lower', cmap='YlOrRd')
    ax1.set_title('Chemical Gradient (darker = stronger)')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')

    ax1.plot(world.food_pos[0], world.food_pos[1],
             'g*', markersize=15, label='Food')

    for agent in world.agents:
        ax1.plot(agent.pos[0], agent.pos[1],
                'b^', markersize=10, label='Worm')

    ax1.legend()

    positions = []
    for _ in range(50):
        for agent in world.agents:
            positions.append(agent.pos)
        world.step()

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    ax2.imshow(world.gradient.T, origin='lower',
               cmap='YlOrRd', alpha=0.3)
    ax2.plot(xs, ys, 'b-', alpha=0.5, linewidth=1)
    ax2.plot(xs[0], ys[0], 'b^', markersize=10, label='Start')
    ax2.plot(world.food_pos[0], world.food_pos[1],
             'g*', markersize=15, label='Food')
    ax2.set_title('Random Worm Movement (50 steps)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('worm_world_visualization.png', dpi=100)
    plt.show()
    print("Saved visualization to worm_world_visualization.png")

if __name__ == "__main__":
    print("Creating world...")
    world = WormWorld(width=20, height=20)
    print(f"World created: {world.width}x{world.height} grid")
    print(f"Food position: {world.food_pos}")

    for agent in world.agents:
        print(f"Worm starting position: {agent.pos}")
        print(f"Gradient at worm start: {world.gradient[1][1]:.4f}")

    print(f"Gradient at food: {world.gradient[10][10]:.4f}")
    print("\nVisualizing...")
    visualize_world(world)
    
