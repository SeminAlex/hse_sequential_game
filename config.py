#### SELF PLAY
EPISODES = 30
MCTS_SIMS = 50
MEMORY_SIZE = 30000
TURNS_UNTIL_TAU0 = 10  # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

#### RETRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_CNN_LAYERS = [
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
    {'filters': 75, 'kernel_size': (4, 4)},
]

#### EVALUATION
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

run_folder = './run/'
run_archive_folder = './run_archive/'

# Units database, information taken from
# http://ru.mightandmagic.wikia.com/wiki/Heroes_of_Might_and_Magic_V
# key - unit name, value - list(attack, defense, damage, life, initiative, speed, range)
Units = {
    "Peasant": [1, 1, 1, 3, 8, 4, 1],
    "Archer": [4, 3, 2, 7, 9, 4, 10],
    "Marksman": [4, 4, 2, 10, 8, 4, 12],
    "Footman": [4, 8, 2, 16, 8, 4, 1],
    "Squire": [5, 9, 2, 26, 8, 4, 1],
    "Griffin": [7, 5, 5, 30, 15, 7, 1],
    "Imperial griffin": [9, 8, 5, 35, 15, 7, 1],
    "Priest": [12, 12, 9, 54, 10, 5, 7],
    "Inquisitor": [16, 16, 9, 80, 10, 5, 7],
    "Angel": [27, 27, 45, 180, 11, 6, 1],
    "Archangel": [31, 31, 50, 220, 11, 8, 1],
    "Gremlin": [2, 2, 1, 5, 7, 3, 5],
    "Master gremlin": [2, 2, 1, 6, 11, 5, 7],
    "Stone gargoyle": [3, 4, 1, 15, 9, 6, 1],
    "Obsidian gargoyle": [3, 5, 1, 20, 10, 7, 1],
    "Iron golem": [5, 5, 3, 18, 7, 4, 1],
    "Steel golem": [6, 6, 5, 24, 7, 4, 1],
    "Mage": [10, 10, 7, 18, 10, 4, 3],
    "Archmage": [10, 10, 7, 30, 10, 4, 4],
    "Djinn": [13, 12, 12, 33, 12, 7, 1],
    "Djinn sultan": [14, 14, 14, 45, 12, 8, 1],
    "Rakshasa rani": [25, 20, 15, 120, 9, 5, 1],
    "Rakshasa raja": [25, 20, 23, 140, 8, 6, 1],
    "Colossus": [27, 27, 40, 175, 10, 6, 1],
    "Titan": [30, 30, 40, 190, 10, 6, 5],
    "Imp": [2, 1, 1, 4, 13, 5, 1],
    "Familiar": [3, 2, 2, 6, 13, 5, 1],
    "Horned demon": [1, 3, 1, 13, 7, 5, 1],
    "Horned overseer": [3, 4, 1, 13, 8, 5, 1],
    "Hell hound": [4, 3, 3, 15, 13, 7, 1],
    "Cerberus": [4, 2, 4, 15, 13, 8, 1],
    "Succubus": [6, 6, 6, 20, 10, 4, 6],
    "Succubus mistress": [6, 6, 6, 30, 10, 4, 6],
    "Hell charger": [13, 13, 8, 50, 16, 7, 1],
    "Nightmare": [18, 18, 8, 66, 16, 8, 1],
    "Pit fiend": [21, 21, 13, 110, 8, 4, 1],
    "Pit lord": [22, 21, 13, 120, 8, 4, 1],
    "Devil": [27, 25, 36, 166, 11, 7, 1],
    "Arch devil": [32, 29, 36, 199, 11, 7, 1],
}

attributes = list(["attack", "defense", "damage", "life", "initiative", "speed", "range"])
name2index = dict(map(lambda x: (x[1], x[0]), enumerate(Units.keys())))
index2name = dict(enumerate(Units.keys()))

for i in Units:
    if len(Units[i]) != len(attributes):
        print("Unit - {}, length = {}".format(i, len(Units[i])))
