import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from collections import deque
import random
import hashlib

class DQN:
    def __init__(self, state_shape, action_size, replay_buffer_size=10000, batch_size=128, gamma=0.99, lr=0.001):
        self.action_size = action_size
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.gamma = gamma
        self.lr = lr
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.hashtable = [[{} for _ in range(10)] for _ in range(10)] 
        self.maxBufferSize = replay_buffer_size
        self.replay_buffer = deque(maxlen=self.maxBufferSize)
    
    def _build_model(self):
        inputs = layers.Input(shape=self.state_shape)
        x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # Residual block 1
        for _ in range(3):
            shortcut = x
            x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)
        
        # Residual block 2
        for i in range(4):
            shortcut = x
            if i == 0:  # 첫 번째 블록에서만 채널 수가 변경되므로, 이를 맞추기 위해 1x1 Convolution 사용
                shortcut = layers.Conv2D(128, kernel_size=1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, kernel_size=3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)

        # Global average pooling and output
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = [self.replay_buffer.popleft() for _ in range(self.batch_size)]
        minibatch = random.sample(minibatch, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        valid_actions  =[i[5] for i in minibatch]
        target = self.model.predict(next_states.reshape(-1,8,8,1),verbose=None)
        target_next = self.target_model.predict(next_states.reshape(-1,8,8,1),verbose=None)
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i][valid_actions[i]])
        loss = self.model.train_on_batch(states.reshape(-1,8,8,1), target)
        return loss  # 손실 값 반환
    

    def BehaviorPolicy(self, state,valid_action):
        q_values = self.model.predict(state.reshape(-1,8,8,1),verbose=None)[0][valid_action]
        count = []
        for i in range(len(valid_action)):
            count.append(self.GetCount(state,valid_action[i]))
        if(sum(count)==0):
            return random.choice(valid_action)
        else:
            uct_values = q_values+ np.sqrt(2*np.log(np.sum(count))/(1+np.array(count)))
            return valid_action[np.argmax(uct_values)]
    
    def EstimatePolicy(self, state , valid_action):
        q_values = self.target_model.predict(state.reshape(-1,8,8,1),verbose=None)[0]
        mask = np.zeros_like(q_values, dtype=bool)
        mask[valid_action] = True
        return valid_action[np.argmax(q_values[mask])]
    
    def load(self, name):
        self.model.load_weights(name)
    
    def save(self, name):
        self.model.save_weights(name)

    def InsertBuffer(self, state, action, reward,next_states,done,valid_actions):
        self._InsertHashTable(state,action)
        if len(self.replay_buffer) > self.maxBufferSize:
            self.replay_buffer.popleft() 
        self.replay_buffer.append([state, action, reward,next_states,done,valid_actions])

    def GetCount(self, state,action):
        hash_value = self._Gethash(state,action)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % 10
        col = int(second_half, 16) % 10
        if hash_value not in self.hashtable[row][col]:
            return 0
        else:
            return self.hashtable[row][col][hash_value]

    def flush(self):
        buffer = self.replay_buffer
        self.replay_buffer = []
        return buffer

    def _Gethash(self, state,action):
        # 2차원 배열의 각 요소 값을 결합하여 하나의 문자열로 만듭니다.
        combined_string = ''.join(map(str, state))+ str(action)
        # 결합된 문자열에 대한 해시 값을 계산합니다.
        hash_value = hashlib.md5(combined_string.encode()).hexdigest()
        return hash_value

    def _InsertHashTable(self, state,action):
        hash_value = self._Gethash(state,action)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % 10
        col = int(second_half, 16) % 10
        if hash_value not in self.hashtable[row][col]:
            self.hashtable[row][col][hash_value] = 1
        else:
            self.hashtable[row][col][hash_value] += 1