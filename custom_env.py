import numpy as np
import math

class CustomEnv:

    def __init__(self):
        print("<---> New environment initialized")
        #######################################
        # Environment variables ###############
        #######################################
        self.MAX_EPISODES = 50
        self.MAX_STEPS = 50
        self.INPUT_SHAPE = (7, 2)
        self.OUTPUT_SHAPE = (81)
        self.TRAIN_CAPACITY = 174.5 * 4 #   Kt = St_confort*densite_max_t (unit: passengers)
        self.PLATFORM_CAPACITY = 270 * 4 #   Kp_opt = Sp*densite_max_opt (unit: passengers)
        self.BLOCK_LENGTH = 200
        #######################################

        #######################################
        # Action related constraints: ########
        #######################################
        # Number of trains Range
        self.MIN_NBR_TRAINS = 20
        self.MAX_NBR_TRAINS = 148  # Nb - 2 = 150 - 2

        # V MAX Range [MIN_V_MAX, MAX_V_MAX] = [1, 22]    =>  Controlling Headway
        self.MIN_V_MAX = 5  # m/s
        self.MAX_V_MAX = 22  # m/s

        # MAX Dwell time range  =>  Controlling Dwell time
        self.MIN_MAX_DWELL = 16  # s
        self.MAX_MAX_DWELL = 45  # s

        # Optimal density (platform's capacity)  =>   Controlling I flow matrix of passengers to the platform
        self.MIN_DENSITY_MAX_OPT = 2  # pass / m2
        self.MAX_DENSITY_MAX_OPT = 5  # pass / m2
        #######################################

        #######################################
        # Actions #############################
        #######################################
        #   0      :   Decrease value
        #   1       :   Stay same
        #   2       :   Increase value
        self.DECREASE = 0
        self.STAY = 1
        self.INCREASE = 2
        self.CHANGES = (self.DECREASE, self.STAY, self.INCREASE)

        self.ACTIONS = np.zeros(self.OUTPUT_SHAPE)
        for i in range(self.OUTPUT_SHAPE):
            self.ACTIONS[i] = i

        self.MAPPING = {}
        i = 0
        for j in range(len(self.CHANGES)):
            for k in range(len(self.CHANGES)):
                for l in range(len(self.CHANGES)):
                    for m in range(len(self.CHANGES)):
                        self.MAPPING[self.ACTIONS[i]] = [self.CHANGES[j],
                                                         self.CHANGES[k],
                                                         self.CHANGES[l],
                                                         self.CHANGES[m]]
                        i += 1

        self.NBR_TRAINS_INDEX = 0
        self.DENSITY_MAX_OPT_INDEX = 1
        self.V_MAX_INDEX = 2
        self.DWELL_TIME_INDEX = 3
        #######################################

        #######################################
        # Factors of each action ##############
        #######################################
        # Example: new value = actual value + (ACTION * FACTOR)
        #          nbr trains = 5, action = decrease, factor = 1
        #          ==>  new nbr trains = 5 - 1 = 4
        self.FACTOR_NBR_TRAINS = 1
        self.FACTOR_V_MAX = 0.2
        self.FACTOR_MAX_DWELL = 0.2
        self.FACTOR_KP_OPT = 0.02
        #######################################

        #######################################
        # Environment related #################
        #######################################
        self.currentState = np.zeros(self.INPUT_SHAPE)
        #######################################

    def reset(self):
        print("----> Environment reset")
        self.currentState = np.zeros(self.INPUT_SHAPE)
        return self.currentState

    def print_action(self, action):
        if action is not None:
            if action == self.DECREASE:
                print("### Choosing Decrease")
            elif action == self.INCREASE:
                print("### Choosing Increase")
            else:
                print("### Choosing Same")

    def print(self):
        print("----> Environment")
        print("MAX_EPISODES = " + str(self.MAX_EPISODES))
        print("MAX_STEPS = " + str(self.MAX_STEPS))
        print("INPUT_SHAPE = " + str(self.INPUT_SHAPE))
        print("OUTPUT_SHAPE = " + str(self.OUTPUT_SHAPE))
        print("MIN_NBR_TRAINS = " + str(self.MIN_NBR_TRAINS))
        print("MAX_NBR_TRAINS = " + str(self.MAX_NBR_TRAINS))
        print("MIN_V_MAX = " + str(self.MIN_V_MAX))
        print("MAX_V_MAX = " + str(self.MAX_V_MAX))
        print("MIN_MAX_DWELL = " + str(self.MIN_MAX_DWELL))
        print("MAX_MAX_DWELL = " + str(self.MAX_MAX_DWELL))
        print("MIN_DENSITY_MAX_OPT = " + str(self.MIN_DENSITY_MAX_OPT))
        print("MAX_DENSITY_MAX_OPT = " + str(self.MAX_DENSITY_MAX_OPT))
        print("DECREASE = " + str(self.DECREASE))
        print("STAY = " + str(self.STAY))
        print("INCREASE = " + str(self.INCREASE))
        print("ACTIONS = " + str(self.ACTIONS))
        print("FACTOR_NBR_TRAINS = " + str(self.FACTOR_NBR_TRAINS))
        print("FACTOR_V_MAX = " + str(self.FACTOR_V_MAX))
        print("FACTOR_MAX_DWELL = " + str(self.FACTOR_MAX_DWELL))
        print("FACTOR_KP_OPT = " + str(self.FACTOR_KP_OPT))
        print("<---- Environment")

    def convert_to_dqn_array(self, current_state):
        # Current state:
        # A dictionary of 14 value for 7 variables (mean and std for each)

        dqn_input = np.zeros(self.INPUT_SHAPE)
        # DQN Input:
        # Array of 7 x 2
        #                       0           1
        #                       Mean        STD
        #                   -------------------------
        #    0  h_out       |           |           |
        #                   -------------------------
        #    1  I           |           |           |
        #                   -------------------------
        #    2  A           |           |           |
        #                   -------------------------
        #    3  mu          |           |           |
        #                   -------------------------
        #    4  Q           |           |           |
        #                   -------------------------
        #    5  P           |           |           |
        #                   -------------------------
        #    6  dwell       |           |           |
        #                   -------------------------
        dqn_input[0, 0] = current_state["h_out_mean"]
        dqn_input[0, 1] = current_state["h_out_std"]
        dqn_input[1, 0] = current_state["I_mean"]
        dqn_input[1, 1] = current_state["I_std"]
        dqn_input[2, 0] = current_state["A_mean"]
        dqn_input[2, 1] = current_state["A_std"]
        dqn_input[3, 0] = current_state["mu_mean"]
        dqn_input[3, 1] = current_state["mu_std"]
        dqn_input[4, 0] = current_state["Q_mean"]
        dqn_input[4, 1] = current_state["Q_std"]
        dqn_input[5, 0] = current_state["P_mean"]
        dqn_input[5, 1] = current_state["P_std"]
        dqn_input[6, 0] = current_state["dwell_mean"]
        dqn_input[6, 1] = current_state["dwell_std"]
        return dqn_input

    def step(self, octave_adapter, current_state, nbr_trains, v_max, max_dwell, density_max_opt, action):
        action_mapping = self.MAPPING[self.ACTIONS[action]]

        current_v_max = v_max

        nbr_trains = self.apply_action_to_nbr_trains(action_mapping, nbr_trains)
        v_max = self.apply_action_to_v_max(action_mapping, v_max)
        density_max_opt = self.apply_action_to_density_max_opt(action_mapping, density_max_opt)
        max_dwell = self.apply_action_to_max_dwell(action_mapping, max_dwell)
        octave_adapter.write_to_octave(nbr_trains, v_max, max_dwell, density_max_opt)
        octave_adapter.run_octave()
        new_state = octave_adapter.read_from_octave()

        reward = self.calculate_reward(current_state, new_state, v_max, current_v_max)

        new_params = {
            'nbr_trains': nbr_trains,
            'density_max_opt': density_max_opt,
            'v_max': v_max,
            'max_dwell': max_dwell
        }
        return (new_state, new_params, reward, False, "")

    def calculate_reward(self, current_state, new_state, new_v_max, current_v_max):
        # Rewards
        print("----> Calculating reward")
        # For very low used capacity, negative value
        # For moderate used capacity (near 70%) max value
        # For highly used capacity, positive but low value
        used_capacity = (new_state["P_mean"] / self.TRAIN_CAPACITY)
        reward_comfort = math.sin((used_capacity - 0.2) * math.pi) - 0.5
        # 600 seconds (10 minutes of headway), so as long as the headway less than 5 minutes, the reward is positive
        reward_headway = (600 - new_state["h_out_mean"])/1000
        # 0.5 means
        reward_waiting = 0.5 - (new_state["Q_mean"])/self.PLATFORM_CAPACITY
        #if reward_headway < 0 and reward_waiting > 0:
        #    reward_waiting = -1 * reward_waiting
        # reward_vmax = new_v_max - current_v_max
        print("<-----> Calculating reward: reward_comfort = " + str(reward_comfort) + " for used (" + str(used_capacity) + ")")
        print("<-----> Calculating reward: reward_headway = " + str(reward_headway) + " for hout (" + str(new_state["h_out_mean"]) + ")")
        print("<-----> Calculating reward: reward_waiting = " + str(reward_waiting))

        # Alphas (Weights of each individual reward)
        alpha_comfort = 4
        alpha_headway = 10
        alpha_waiting = 0.25
        # alpha_vmax = 1
        reward = alpha_comfort * reward_comfort + alpha_headway * reward_headway + alpha_waiting * reward_waiting
        print("<---- Calculating reward:", str(reward))
        return reward

    def apply_action_to_nbr_trains(self, action_mapping, nbr_trains):
        # NBR trains
        if action_mapping[self.NBR_TRAINS_INDEX] == self.DECREASE:
            if nbr_trains > self.MIN_NBR_TRAINS:
                nbr_trains -= self.FACTOR_NBR_TRAINS
        if action_mapping[self.NBR_TRAINS_INDEX] == self.INCREASE:
            if nbr_trains < self.MAX_NBR_TRAINS:
                nbr_trains += self.FACTOR_NBR_TRAINS
        return nbr_trains

    def apply_action_to_density_max_opt(self, action_mapping, density_max_opt):
        #  Density max Opt
        if action_mapping[self.DENSITY_MAX_OPT_INDEX] == self.DECREASE:
            if density_max_opt >= self.MIN_DENSITY_MAX_OPT + self.FACTOR_KP_OPT:
                density_max_opt -= self.FACTOR_KP_OPT
        if action_mapping[self.DENSITY_MAX_OPT_INDEX] == self.INCREASE:
            if density_max_opt <= self.MAX_DENSITY_MAX_OPT - self.FACTOR_KP_OPT:
                density_max_opt += self.FACTOR_KP_OPT
        return density_max_opt

    def apply_action_to_v_max(self, action_mapping, v_max):
        # V max
        if action_mapping[self.V_MAX_INDEX] == self.DECREASE:
            if v_max >= self.MIN_V_MAX + self.FACTOR_V_MAX:
                v_max -= self.FACTOR_V_MAX
        if action_mapping[self.V_MAX_INDEX] == self.INCREASE:
            if v_max <= self.MAX_V_MAX - self.FACTOR_V_MAX:
                v_max += self.FACTOR_V_MAX
        return v_max

    def apply_action_to_max_dwell(self, action_mapping, max_dwell):
        # Dwell time
        if action_mapping[self.DWELL_TIME_INDEX] == self.DECREASE:
            if max_dwell >= self.MIN_MAX_DWELL + self.FACTOR_MAX_DWELL:
                max_dwell -= self.FACTOR_MAX_DWELL
        if action_mapping[self.DWELL_TIME_INDEX] == self.INCREASE:
            if max_dwell <= self.MAX_MAX_DWELL - self.FACTOR_MAX_DWELL:
                max_dwell += self.FACTOR_MAX_DWELL
        return max_dwell



