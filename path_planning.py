from env import PathPlanningEnv

def main():
    env = PathPlanningEnv("grid1.bmp")
    state = env.reset((10, 10), (3, 3))
    while True:
        print(state)
        action = int(input())
        state, reward, terminal = env.step(action)
        print(reward, terminal)

if __name__=='__main__':
    main()