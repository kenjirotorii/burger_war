import numpy as np

import DQN

my_pos = [
    [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [7, 7], [8, 9], [8, 10], [7, 11], [6, 11], [5, 12], [4, 12], [3, 13], [2, 13]
]

enemy_pos = [
    [14, 14], [13, 13], [12, 13], [11, 13], [10, 13], [9, 13], [8, 13], [7, 13], [6, 13], [5, 13], [4, 13], [3, 13], [2, 13], [1, 13]
]

my_angle = [
    45, 45, 45, 45, 45, 70, 70, 90, 135, 180, 135, 180, 135, 180
]

# 自分が向いている向きを２次元ベクトル(n*n)にして返す
def get_ang_matrix(angle, n=16):
    while angle > 0 : angle -= 360
    while angle < 0 : angle += 360
    my_ang  = np.zeros([n, n])
    for i in range(16):
        for j in range(16):
            if 360-22.5 < angle or angle <= 22.5 :              #   0°
                if 10 <= i and 10 <= j      : my_ang[i][j] = 1
            if  45-22.5 < angle <=  45+22.5 :                   #  45°
                if 10 <= i and  5 <= j <= 10: my_ang[i][j] = 1
            if  90-22.5 < angle <=  90+22.5 :                   #  90°
                if 10 <= i and  5 >= j      : my_ang[i][j] = 1
            if 135-22.5 < angle <= 135+22.5 :                   # 135°
                if  5 <= i <=10 and  5 >= j : my_ang[i][j] = 1
            if 180-22.5 < angle <= 180+22.5 :                   # 180°
                if  5 >= i and  5 >= j      : my_ang[i][j] = 1
            if 225-22.5 < angle <= 225+22.5 :                   # 225°
                if  5 >= i and  5 <= j <= 10: my_ang[i][j] = 1
            if 270-22.5 < angle <= 270+22.5 :                   # 270°
                if  5 >= i and  10 <= j     : my_ang[i][j] = 1
            if 315-22.5 < angle <= 315+22.5 :                   # 315°
                if  5 <= i <=10 and 10 <= j : my_ang[i][j] = 1
    #print(my_ang)
    return my_ang

actions = [
    [2, 2], [3, 3], [4, 4], [5, 5], [7, 7], [8, 9], [8, 10], [7, 11], [6, 11], [5, 12], [4, 12], [3, 13], [2, 13]
]
    
NUM_STEP = 13

if __name__ == "__main__":
    
    mainQN = DQN.QNetwork(debug_log=True)
    memory = DQN.Memory(max_size=1000)
    
    for i in range(NUM_STEP):
        state = np.zeros((16, 16, 8))
        state[my_pos[i] + [0]] = 1.0
        state[enemy_pos[i] + [1]] = 1.0
        state[:, :, 2] = get_ang_matrix(my_angle[i])
        state = state.reshape((1, 16, 16, 8))

        action = actions[i]

        if i != NUM_STEP - 1:
            next_state = np.zeros((16, 16, 8))
            next_state[my_pos[i + 1] + [0]] = 1.0
            next_state[enemy_pos[i + 1] + [1]] = 1.0
            next_state[:, :, 2] = get_ang_matrix(my_angle[i + 1])
            next_state = state.reshape((1, 16, 16, 8))
            reward = 0.0
        else:
            next_state = np.zeros((1, 16, 16, 8))
            reward = 1.0

        memory.add((state, action, reward, next_state))
        #print(memory.buffer[i])

    print('start learning')
    for epoch in range(10):
        loss = mainQN.replay(memory, 5, 0.97)
        pred = mainQN.model.predict(memory.buffer[NUM_STEP - 2][0])
        print('epoch:{}, loss:{}'.format(epoch, loss))
        print(pred.shape)