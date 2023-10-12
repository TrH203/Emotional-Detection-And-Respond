import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Define the true number you want to animate to
true_number = 42

# Create a figure and axis for the bar chart
fig, ax = plt.subplots()

# Create a bar with an initial height of 0
bar = ax.bar(0, 0, width=0.5, align='center')

# Function to update the animation frame
def update(frame):
    # Calculate the new bar height for the animation frame
    new_height = frame / true_number
    # Update the bar height
    bar[0].set_height(new_height)

# Create an animation with FuncAnimation
animation = FuncAnimation(fig, update, frames=np.arange(0, true_number + 1), repeat=False, interval=50)

# Set the y-axis limits to make the animation smooth from below
ax.set_ylim(0, 1)

# Set axis labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Percentage')
ax.set_title('Smooth Bar Animation')

# Show the animation
plt.show()
