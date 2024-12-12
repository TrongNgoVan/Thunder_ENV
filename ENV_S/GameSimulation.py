from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from CustomAgent import Agent

env = TetrisDoubleEnv()

done = False
state = env.reset()
agent_list = [Agent(), Agent()]

while not done:
    img = env.render(mode='rgb_array') # img is rgb array, you need to render this or can check my colab notebook in readme file
    action = agent_list[env.game_interface.getCurrentPlayerID()].choose_action(state)
    state, reward, done, _ = env.step(action)
    env.take_turns()
