import matplotlib.pyplot as plt
from random import choice


class RandomWalk:
    def __init__(self, num_limit):
        self.num_limit = num_limit
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):

        while len(self.x_values) < self.num_limit:
            x_dir = choice([-1, 1])
            x_step = choice([0, 1, 2, 3])
            y_dir = choice([-1, 1])
            y_step = choice([0, 1, 2, 3])

            if x_step == 0 and y_step == 0:
                continue

            x = self.x_values[-1] + x_dir * x_step
            y = self.y_values[-1] + y_dir * y_step
            self.x_values.append(x)
            self.y_values.append(y)

# input = range(1, 1001)
# squares = [x**2 for x in input]



while True:
    random_walk = RandomWalk(50_000)
    random_walk.fill_walk()

    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 9))
    # ax.plot(input, squares, linewidth=3)
    ax.set_title('Squares', fontsize=24)
    ax.set_xlabel('Value', fontsize=14)
    ax.set_ylabel('Square of Value', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    point_num = range(random_walk.num_limit)
    ax.scatter(random_walk.x_values, random_walk.y_values, c=point_num, cmap=plt.cm.Blues, edgecolors='none', s=10)
    ax.scatter(random_walk.x_values[-1], random_walk.y_values[-1], c='green', edgecolors='none', s=100)
    ax.scatter(random_walk.x_values[0], random_walk.y_values[0], c='red', edgecolors='none', s=100)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #ax.axis([0, 1100, 0, 1100000])
    #plt.savefig('squares_plot.png', bbox_inches='tight')
    plt.show()

    keep_running = input('make another walk?(y/n): ')
    if keep_running == 'n':
        break
