import hashlib
class DQN:
    #state,action,reward,count 
    replayBuffer : list
    maxBufferSize = 100
    #3d hashtable
    hashtable : list
    hashNum = 10 
    def __init__(self) -> None:
        self.hashtable = [[{} for _ in range(self.hashNum)] for _ in range(self.hashNum)] 
        self.replayBuffer = []
        pass

    
    def train(self):
        pass


    def InsertBuffer(self,state,action,reward):
        self._InsertHashTable(state)
        if( len(self.replayBuffer) < self.maxBufferSize):
            self.replayBuffer.append([state,action,reward])
            return True
        return False
    def GetCount(self,state):
        hash_value= self._hash_2d_array(state)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % self.hashNum
        col = int(second_half, 16)  % self.hashNum
        if hash_value not in self.hashtable[row][col]:
            return 0
        else:
            return self.hashtable[row][col][hash_value]

    def flush(self):
        buffer = self.replayBuffer
        self.replayBuffer = []
        return buffer
    def _hash_2d_array(self,arr):
        # 2차원 배열의 각 요소 값을 결합하여 하나의 문자열로 만듭니다.
        combined_string = ''.join(map(str, arr))
        # 결합된 문자열에 대한 해시 값을 계산합니다.
        hash_value = hashlib.sha256(combined_string.encode()).hexdigest()
        return hash_value
    def _InsertHashTable(self,state):
        hash_value= self._hash_2d_array(state)
        half_length = len(hash_value) // 2
        first_half = hash_value[:half_length]
        second_half = hash_value[half_length:]
        row = int(first_half, 16) % self.hashNum
        col = int(second_half, 16)  % self.hashNum
        if hash_value not in self.hashtable[row][col]:
            self.hashtable[row][col][hash_value]  = 1
        else:
            self.hashtable[row][col][hash_value]  += 1