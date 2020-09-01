def reward_fn(state, action):
    xpen = np.clip(-(state[3] - .15)**2, -1, 0)
    #xpen = 0.0

    ypen = np.clip(-(state[4] - 1.2)**2, -4, 0)
    #ypen = 0.0

    alive = 5.0
    return xpen + ypen + alive
