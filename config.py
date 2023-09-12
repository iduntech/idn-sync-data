DEVICE = "MBT"  # 'MBT'
FILTER_RANGE = [20, 40]
HIGHPASS_FREQ = 0.3
BASE_SAMPLE_RATE = 250
IDUN_SAMPLE_RATE = 250

FIRST_LAG_EPOCH_SIZE = BASE_SAMPLE_RATE * 60
SECOND_LAG_EPOCH_SIZE = BASE_SAMPLE_RATE * 30  # 30 works best
TOTAL_LINEAR_SEGMENTS = 5000
POLYNOMIAL_ORDER = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CUT = "end"  #'equal'
DISCONTINUITY_THRESHOLD = 20

# Prodigy configuration (PRODIGY)
PRODIGY_CHANNEL_1 = "LEFT_EYE"
PRODIGY_CHANNEL_2 = "RIGHT_EYE"
PRODIGY_SAMPLE_RATE = 120
PRODIGY_SCALE_FACTOR = 1000000

# full scalp configuration (MBT)
MBT_CHANNEL_1 = "T8"
MBT_CHANNEL_2 = "T7"
MBT_SAMPLE_RATE = 250
MBT_SCALE_FACTOR = 1
