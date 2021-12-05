from random import randint
from plotly.graph_objs import Bar, Layout
from plotly import offline

class Die:
    def __init__(self, num_side=6):
        self.num_side = num_side

    def roll(self):
        return randint(1, self.num_side)


die1 = Die()
die2 = Die()

results = []
for roll in range(1000):
    result = die1.roll() + die2.roll()
    results.append(result)

frequencies = []
for i in range(2, 13):
    frequency = results.count(i)
    frequencies.append(frequency)

print(frequencies)

x_values = list(range(2, 13))
data = [Bar(x=x_values, y=frequencies)]

x_axis_config = {'title': 'result', 'dtick': 1}
y_axis_config = {'title': 'frequency of result'}

my_layout = Layout(title='roll 2 D6 1000 times result',
                   xaxis=x_axis_config, yaxis=y_axis_config)
offline.plot({'data': data, 'layout': my_layout}, filename='d6_d6.html')