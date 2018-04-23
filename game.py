from enum import Enum
from copy import deepcopy as cp
from random import randrange, sample
from collections import deque
import config as cfg


def _create_zero_line(length):
    return [0 for _ in range(length)]


def sign(x): return 1 if x >= 0 else -1


class Status(Enum):
    win = 1
    lose = -1
    draw = 0
    not_ended = -2


class Game:
    __slots__ = ["field_", "player_", "units_", "height_", "width_", "FEATURE_NUM", "status", ]
    """
    field_ - matrix: FEATURE_NUM x (height*width), where each row has length= height*width
              field_[0-(FEATURE_NUM-1)] - integer arrays, K in raw i means, that unit, which name is unit_names[i] is 
                                          located in this cell, its total life is K and it is all units of FIRST player
              field_[FEATURE_NUM-(FEATURE_NUM*2-1)] - integer arrays, same as previous but for SECOND player
              field_[FEATURE_NUM*2]   - binary array, 1 means that current unit can move into this location
              field_[FEATURE_NUM*2+j] - binary array, 1 means that current unit can attack enemy in this location if he 
                                            current unit made movements in location 'j-1'


    player_       - binary variable that indicate which turn is now
    height_       - int variable, shows height of field
    width_        - int variable, shows width of field
    FEATURE_NUM   - int variable, total number of all possible units
    status        - Status variable, can be any value from 'Status' class
    units_        - (Name, Owner, Raw, Location) array which determinate queue of units turns
    """

    def __init__(self, height, weight, units=None, field=None):
        self.height_ = height
        self.width_ = weight
        self.FEATURE_NUM = len(cfg.name2index)
        self.status = Status.not_ended
        if field:
            self.field_ = cp(field)
        else:
            self.__generate_field(units)
        self.units_ = deque(*self.units_sort_())

    def __generate_field(self, units=None):
        length = self.width_ * self.height_
        self.field_ = [_create_zero_line(length) for _ in range(self.FEATURE_NUM * 2 + 3)]

        if units:
            if len(units) < 1 or len(units[0]) != 5:
                raise Exception("ERROR: List of units must contain at least one tuple like " +
                                "(X, Y, UNIT_NAME, PLAYER, COUNT)")
            for x, y, unit_name, player, count in units:
                attributes = cfg.Units.get(unit_name, None)
                if not attributes:
                    raise Exception("ERROR: Unit with name {} doesn't exist. Possible unit names : {}"
                                    .format(unit_name, cfg.Units.keys()))
                raw = cfg.name2index[unit_name] + self.FEATURE_NUM * player
                coordinate = x * self.width_ + y
                self.field_[raw][coordinate] = cfg.Units[unit_name][3] * count
        return

    def check_status(self):
        """
        Check if game is ended. Return 0 if there are no winner, 1 if first player win, and -1 otherwise
        """
        ifetures = [(0, self.FEATURE_NUM), (self.FEATURE_NUM, self.FEATURE_NUM * 2)]
        player_win = [sum(map(sum, self.field_[start:end])) for start, end in ifetures]
        return 0 if player_win[0] and player_win[1] else -1 if player_win[1] else 1

    def take_action(self, actions):
        """
        
        :param actions: <list>, length = (self.height_ * self.width_) * 2; first half of list is associated with
                                                                            movements, second - with attack
        :return: <Game>, return new State of this game
        """
        actions = [actions[:self.height_ * self.width_], actions[self.height_ * self.width_:]]
        movement = actions[0].index(max(actions[0]))
        current_unit = self.units_[0]  # (Name, Owner, Raw, Location)
        unit_raw = self.field_[current_unit[2]]

        # make move
        if not self.is_free(movement):
            return None, Status.lose
        unit_raw[current_unit[3]], unit_raw[movement] = unit_raw[movement], unit_raw[current_unit[3]]
        # make attack
        defender_index = actions[1].index(max(actions[1]))
        defender = (self.find_unit_raw(defender_index), defender_index)
        att_total, def_total = self.fight(current_unit[2:], defender)
        if att_total < 0 or def_total < 0:
            raise Exception("fight method returned wrong values, att = {}, def = {}".format(att_total, def_total))

        pass

    def is_free(self, location):
        for i in range(self.FEATURE_NUM * 2):
            if self.field_[i][location] != 0:
                return False
        return True

    def fill_actions(self):
        unit = self.units_[0]
        location = unit[3]
        speed = cfg.Units[unit[0]][5]
        diapason = cfg.Units[unit[0]][6]
        raw, column = self.get_raw_column(location)
        moveRaw = self.FEATURE_NUM * 2

        for i in range(len(self.field_[0])):
            self.field_[moveRaw][i] = 1 if i == location else 0
            raw_i, col_i = self.get_raw_column(i)

            if self.is_free(i) and self.in_moves(raw_i, col_i, raw, column, speed):
                self.field_[moveRaw][i] = 1
        self.field_[moveRaw + 1:] = self.move_attack_matrix()

    def move_attack_matrix(self):
        def raw_creation(line):
            result = [0] * length
            raw = line // self.width_
            column = line - self.width_ * raw
            for i in range(length):
                raw_i = i // self.width_
                col_i = i - self.width_ * raw_i
                if not self.is_free(i):
                    if self.in_range(raw_i, col_i, raw, column, diapason):
                        result[i] = 1
            return result

        unit = self.units_[0]
        speed = cfg.Units[unit[0]][5]
        diapason = cfg.Units[unit[0]][6]
        length = self.height_ * self.width_
        location = unit[3]
        raw, column = self.get_raw_column(location)

        return [raw_creation(i)
                if self.is_free(i) and self.in_moves(*self.get_raw_column(i), raw, column, speed) else [0] * length
                for i in range(length)]

    def get_raw_column(self, location):
        raw = location // self.width_
        column = location - self.width_ * raw
        return raw, column

    @staticmethod
    def in_moves(raw_to, col_to, raw, column, speed):
        if raw - speed <= raw_to < raw + speed and column - speed <= col_to < column + speed:
            return True
        return False

    @staticmethod
    def in_range(raw_to, col_to, raw, column, diapason):
        if raw - diapason <= raw_to < raw + diapason and column - diapason <= col_to < column + diapason:
            return True
        return False

    @staticmethod
    def make_move(raw, current, new_location):
        raw[current], raw[new_location] = raw[new_location], raw[current]
        pass

    def find_unit_raw(self, column):
        for raw in range(self.FEATURE_NUM * 2):
            if self.field_[raw][column]:
                return raw
        return False

    def units_sort_(self):
        """ return list of tuples (Unit Name, Owner, raw, location)"""
        units_in_game = [(cfg.index2name[i % self.FEATURE_NUM], i // self.FEATURE_NUM, i, j)
                         for i in range(self.FEATURE_NUM * 2) for j in range(len(self.field_[i])) if self.field_[i][j]]
        return sorted(units_in_game, key=lambda x: cfg.Units[x[0]][-3], reverse=True), len(units_in_game)

    @staticmethod
    def calculate_damage(attacker, att_count, defender):
        """
        calculate damage that unit 'attacker' inflict to 'defender'
        :param attacker: <int> index of unit that attack
        :param att_count: <int> unit count in attack army
        :param defender: <int> index of unit that defence
        :return:
        """
        attacker_name = cfg.index2name[attacker]
        defender_name = cfg.index2name[defender]
        damage = cfg.Units[attacker_name][2]  # get damage
        odds = cfg.Units[attacker_name][0] - cfg.Units[defender_name][1]  # difference between attack and defend
        damage_coeff = (1.0 + 0.1 * sign(odds)) ** abs(odds)  # coefficient of damage
        return damage * att_count * damage_coeff

    def get_count(self, unit):
        """
        :param unit:(Unit name, Owner, Raw, Location)
        :return: count of unit at this location
        """
        return self.field_[unit[2]][unit[3]] / cfg.Units[unit[0]]

    def fight(self, attacker, defender):
        """
        Emulate fight between units 'attacker' and 'defender'
        :param attacker: (<int> raw, <int> location), number of unit who attack and his position in the field
        :param defender: (<int> raw, <int> location), number of unit who defends and his position in the field
        :return: (<int>: 'attacker' unit total life after attack, <int>: 'defender' unit total life after attack )
        """
        att_count = self.get_count(self.units_[0])
        def_count = self.get_count((cfg.index2name[defender[0]], None, defender[0], defender[1]))
        # calculate total damage that unit 'attacker' inflict to 'defender'
        damage = self.calculate_damage(attacker[0], att_count, defender[0])
        damage = damage if damage > 1 else 1
        att_total = self.field_[defender[0]][defender[1]] - damage
        att_total = att_total if att_total > 0 else 0

        # calculate total damage that unit 'defender' inflict to 'attacker'
        def_total = 0
        if att_total != 0:
            damage = self.calculate_damage(defender[0], def_count, attacker[0])
            damage = damage if damage > 1 else 1
            def_total = self.field_[attacker[0]][attacker[1]] - damage
            def_total = def_total if def_total > 0 else 0

        return att_total, def_total

    def id(self):
        id = str(self.FEATURE_NUM) + "&&" + str()
        ";".join([",".join(map(str, line)) for line in self.field_])


def genereate_units_array(number, player_percent, weight, height):
    def indext2coordinate(index):
        w = index // weight
        h = index - weight * w
        return w, h

    coord_inexes = sample(range(weight * height), number)
    coordinates = [indext2coordinate(i) for i in coord_inexes]
    ownership = [1 if i < number * player_percent else 0 for i in range(number)]
    counts = [randrange(1, 100) for _ in range(number)]
    units = [(*coordinates[i], *sample(cfg.Units.keys(), 1), ownership[i], counts[i]) for i in range(number)]
    return units


# tmp = genereate_units_array(8, 0.5, 15, 11)
tmp = [(4, 14, 'Rakshasa rani', 1, 27), (7, 12, 'Djinn sultan', 1, 57), (7, 9, 'Pit fiend', 1, 62),
       (9, 6, 'Master gremlin', 1, 39), (5, 11, 'Imperial griffin', 0, 73), (10, 8, 'Steel golem', 0, 55),
       (4, 1, 'Priest', 0, 45), (3, 1, 'Hell charger', 0, 19), (0, 0, 'Peasant', 1, 333)]

print(tmp)
game = Game(weight=15, height=11, units=tmp)
print(game)
print(game.units_sort_())
print(game.units_)
game.units_.append(game.units_[0])
print(game.units_)
game.fill_actions()
print(game.move_attack_matrix())
