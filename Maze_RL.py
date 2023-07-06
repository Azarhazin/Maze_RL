import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt


def evaluate_model(robot, episodes):
    evaluation_moves = []
    for _ in range(episodes):
        maze = Maze()  # Create a new maze for each evaluation episode
        while not maze.is_game_over():
            state, _ = maze.get_state_and_reward()  # Get the current state
            action = robot.choose_action(state, maze.allowed_states[state])  # Choose an action (exploitation)
            maze.update_maze(action)  # Update the maze according to the action
        evaluation_moves.append(maze.steps)  # Get the number of steps taken in this evaluation episode
    return evaluation_moves

if __name__ == '__main__':
    maze = Maze()
    robot = Agent(maze.maze, alpha=0.1, random_factor=0.25)
    moveHistory = []

    for i in range(5000):
        if i % 1000 == 0:
            print(i)

        while not maze.is_game_over():
            state, _ = maze.get_state_and_reward() # get the current state
            action = robot.choose_action(state, maze.allowed_states[state]) # choose an action (explore or exploit)
            maze.update_maze(action) # update the maze according to the action
            state, reward = maze.get_state_and_reward() # get the new state and reward
            robot.update_state_history(state, reward) # update the robot memory with state and reward
            if maze.steps > 1000:
                # end the robot if it takes too long to find the goal
                maze.robot_position = (5, 5)
        
        robot.learn() # robot should learn after every episode
        moveHistory.append(maze.steps) # get a history of number of steps taken to plot later
        maze = Maze() # reinitialize the maze
    # Evaluation
    evaluation_moves = evaluate_model(robot, episodes=10)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.semilogy(moveHistory, "b--", label="Training")
    plt.plot(np.arange(len(moveHistory), len(moveHistory) + len(evaluation_moves)), evaluation_moves, "r-", label="Evaluation")
    plt.xlabel("Episodes")
    plt.ylabel("Number of Steps (log scale)")
    plt.legend()
    plt.show()
