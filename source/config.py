class Config:
    CHUNK_SIZE = 3600
    NEGATIVE_REWARD = -10
    HEX_RADIUS = 200

    class City:
        MELBOURNE = "Melbourne"
        HAMBURG = "Hamburg"

        MAX_X_MEL = 2200
        MAX_Y_MEL = 1600

        DEFAULT_CITY = MELBOURNE

    class Scenario:
        # --- SCENARIO PARAMETERS ---
        # First rain period
        RAIN1_START_TIME = 700
        RAIN1_END_TIME = 900

        # Second rain period
        RAIN2_START_TIME = 1500
        RAIN2_END_TIME = 1700

        # third rain period
        RAIN3_START_TIME = 3000
        RAIN3_END_TIME = 3300

    class Directory:
        ZON_HAM = "./data/hamburg.zon.xml"
        FN_HAM = "./data/hamburg.fn.xml"
        ZON_MEL = "./data/melbourne.zon.xml"
        FN_MEL = "./data/melbourne.fn.xml"

        DEFAULT_ZON = ZON_HAM
        DEFAULT_FN = FN_HAM

    class Paths:
        NoiseConfigsPath = "D:/av_project/NoiseConfigs"
        pklPath = r"D:\av_project\SumoDividedByTime"

    class VehiclesTraffic:
        PROJECT_ROOT = r"D:\av_project"
        NOISE_CONFIGS_PATH = r"D:\av_project\NoiseConfigs"
        PKL_PATH = r"D:\av_project\precalculated_vehicle_traffic.pkl"

    class TrafficCount:
        GreenTraffic = 5
        YellowTraffic = 10
        OrangeTraffic = 15
        RedTraffic = 20
        BlackTraffic = 25

    class SimulatorConfig:
        SIMULATION_START_TIME = 300
        SIMULATION_DURATION = 1300
        TIMEOUT_TIME = 1
        # todo: should change this variable
        BANDWIDTH = 3
        ENABLE_HARD_TASKS = True
        BASELINE_PARALLEL_FREQUENCY = False  # Flag for the parallel frequency baseline
        HARD_TASKS_FREQ_RATIO = 0.6

    class CloudConfig:
        DEFAULT_X = 6000
        DEFAULT_Y = 1500
        DEFAULT_RADIUS = 10000
        # todo: should change this variable
        CLOUD_BANDWIDTH = 1
        MAX_TASK_QUEUE_LEN = 2000
        DEFAULT_COMPUTATION_POWER = 3500
        CLOUD_NODE_FREQUENCY = 5
        CLOSEST_FOG_X = 4214.90
        CLOSEST_FOG_Y = 1932.26
        NUM_CORE = 64


    class FixedFogNodeConfig:
        MAX_TASK_QUEUE_LEN = 400
        DEFAULT_COMPUTATION_POWER = 500
        Fixed_NODE_FREQUENCY = 2
        NUM_CORE = 32

    class MobileFogNodeConfig:
        DEFAULT_RADIUS = 150
        MAX_TASK_QUEUE_LEN = 150
        DEFAULT_COMPUTATION_POWER = 200
        MOBILE_NODE_FREQUENCY = 1.5
        NUM_CORE = 24

    class UserNodeConfig:
        MAX_TASK_QUEUE_LEN = 100
        DEFAULT_COMPUTATION_POWER = 20
        USER_NODE_FREQUENCY = 0.5
        LOCAL_OFFLOAD_POWER_OVERHEAD = 1
        LOCAL_EXECUTE_TIME_OVERHEAD = 1
        NUM_CORE = 8
        HARD_TASK_EXEC_TIME_DIVISOR = 1e6
        HARD_TASK_SPECS = (
            {"period": 7, "size_max": 1200, "cycles_max": 1200},
            {"period": 5, "size_max": 5000, "cycles_max": 1200},
            {"period": 6, "size_max": 1000, "cycles_max": 1000},
        )

    class ZoneManagerConfig:
        ALGORITHM_RANDOM = "Random"
        DEFAULT_ALGORITHM = ALGORITHM_RANDOM

    class NoiseMethod:
        # PROPOSED_METHOD = "Proposed Method"
        # PROPOSED_METHOD2 = "Proposed Method2"
        # PROPOSED_METHOD3 = "Proposed Method3"
        FIRST_CHOICE = "First Choice"
        RANDOM_CHOICE = "Random Choice"
        MIN_DISTANCE = "Min Distance"

        DEFAULT_METHOD = FIRST_CHOICE

    class RandomZoneManagerConfig:
        OFFLOAD_CHANCE: float = 0.5

    class AntennaGain:
        TX: float = 27
        RX: float = -5

    class NoiseConfig:
        T1 = 25
        T2 = 50
        T3 = 75
        NONE = -1

        DEFAULT_THRESHOLD = NONE

    class AttenuationLevel:
        AttenuationLevel1 = [2.3, 2.5, 2.7, 2.9]
        AttenuationLevel2 = [2.4, 2.7, 3, 3.3]
        AttenuationLevel1Name = 1
        AttenuationLevel2Name = 2

        DEFAULT_AttenuationLevel = AttenuationLevel1
        DEFAULT_AttenuationLevelName = AttenuationLevel1Name

    class TrafficNoise:
        TrafficNoiseLevel1 = 0
        TrafficNoiseLevel2 = 1
        DEFAULT_TrafficNoiseLevel = TrafficNoiseLevel1

        class GreenTrafficNoise:
            GreenTrafficNoise1 = [95, 91, 90, 89, 85]
            GreenTrafficNoise2 = [99, 95, 94, 93, 89]

            DEFAULT_GreenTrafficNoise = GreenTrafficNoise1

        class YellowTrafficNoise:
            YellowTrafficNoise1 = [100, 96, 95, 94, 90]
            YellowTrafficNoise2 = [102, 98, 97, 96, 92]

            DEFAULT_YellowTrafficNoise = YellowTrafficNoise1

        class OrangeTrafficNoise:
            OrangeTrafficNoise1 = [105, 101, 100, 99, 95]
            OrangeTrafficNoise2 = [105, 101, 100, 99, 95]

            DEFAULT_OrangeTrafficNoise = OrangeTrafficNoise1

        class RedTrafficNoise:
            RedTrafficNoise1 = [110, 106, 105, 104, 100]
            RedTrafficNoise2 = [108, 104, 103, 102, 98]

            DEFAULT_RedTrafficNoise = RedTrafficNoise1

        class BlackTrafficNoise:
            BlackTrafficNoise1 = [115, 111, 110, 109, 105]
            BlackTrafficNoise2 = [111, 107, 106, 105, 101]

            DEFAULT_BlackTrafficNoise = BlackTrafficNoise1
