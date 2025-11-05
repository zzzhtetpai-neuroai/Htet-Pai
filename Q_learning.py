import numpy as np
#DEFINE CONSTANTS
gamma=0.75
alpha=0.9

#Mapping locations to states
location_to_state={
                    "A":0,"B":1,
                    "C":2,"D":3,
                    "E":4,"F":5,
                    "G":6,"H":7,
                    "I":8,"J":9,
                    "K":10,"L":11,
                    }
#Define actions
actions=[0,1,2,3,4,5,6,7,8,9,10,11]

# Defining the rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])
#Initialize Q-matrix
Q=np.array(np.zeros((12,12)))

#Training the agent
for i in range(1000):
    current_state=np.random.randint(0,12)
    playable_actions=[]
    for j in range(12):
        if R[current_state,j]>0:
            playable_actions.append(j)
    next_state=np.random.choice(playable_actions)
    TD=R[current_state,next_state]+gamma*Q[next_state,np.argmax(Q[next_state,:])]-Q[current_state,next_state]
    Q[current_state,next_state]=Q[current_state,next_state]+alpha*TD


#mapping states to locations
state_to_location={state:location for location,state in location_to_state.items()}



#Function to get the optimal route
def route(start_location,end_location):
    R_new=np.copy(R)
    end_state=location_to_state[end_location]
    R_new[end_state,end_state]=1000
    for i in range(1000):
        current_state=np.random.randint(0,12)
        playable_actions=[]
        for j in range(12):
            if R_new[current_state,j]>0:
                playable_actions.append(j)
        next_state=np.random.choice(playable_actions)
        TD=R_new[current_state,next_state]+gamma*Q[next_state,np.argmax(Q[next_state,])]-Q[current_state,next_state]
        Q[current_state,next_state]=Q[current_state,next_state]+alpha*TD
    route=[start_location]
    next_location=start_location
    while next_location != end_location:
        starting_state=location_to_state[start_location]
        next_state=np.argmax(Q[starting_state,:])
        next_location=state_to_location[next_state]
        route.append(next_location)
        start_location=next_location
    return route

print("Optimal Route:")
print(route("E" ,"G")) 

