from enum import Enum
from copy import deepcopy as cp
from random import randrange, sample
from collections import deque
import numpy as np

import config as cfg


def _create_zero_line(length):
    return [0 for _ in range(length)]


def sign(x): return 1 if x >= 0 else -1


class Status(Enum):
    win = 1
    lose = -1
    not_ended = 0


class GodHand(Enum):
    kill = -1
    move = 0
    add = 1


class Unit:
    __slots__ = ["name", "owner", "raw", "location"]

    def __init__(self, name, owner, raw, location):
        self.name = name
        self.owner = owner
        self.raw = raw
        self.location = location

    @property
    def position(self):
        return self.raw, self.location

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        checker = self.name == other.name and self.owner == self.owner
        checker = checker and self.raw == other.raw and self.location == other.location
        if checker:
            return True
        return False

    def __str__(self):
        return str(list(zip(self.__slots__,[self.name, self.owner,self.raw, self.location])))


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
    units_        - (Name <str>, Owner <int>, Raw <int>, Location <int>) array which determinate queue of units turns
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

    def fill_actions(self):
        """
        Fill sub-matrix self.field_, beginning from self.FEATURE_NUM*2
        self.field_[self.FEATURE_NUM * 2] - binary array,  1 means that current unit can move into this location
        self.field_[FEATURE_NUM*2+j] - binary array, 1 means that current unit can attack enemy in this location if
                                            current unit made movements in location 'j-1'
        """
        unit = self.units_[0]
        location = unit.location
        speed = cfg.Units[unit.name][5]
        diapason = cfg.Units[unit.name][6]
        raw, column = self.get_raw_column(location)
        moveRaw = self.FEATURE_NUM * 2
        raw_length = len(self.field_[0])

        for i in range(raw_length):
            self.field_[moveRaw][i] = 1 if i == location else 0
            raw_i, col_i = self.get_raw_column(i)

            if self.is_free(i) and self.can_move(raw_i, col_i, raw, column, speed):
                self.field_[moveRaw][i] = 1

        # fill attack sub-matrix or create it if it doesn't exist
        if len(self.field_) <= moveRaw + 1:
            self.field_[moveRaw + 1:] = self.move_attack_matrix()
        else:
            for i in range(raw_length):
                new_raw, new_col = self.get_raw_column(i)
                self.field_[i] = [0] * raw_length
                if self.is_free(i) and self.can_move(new_raw, new_col, raw, column, speed):
                    for j in range(raw_length):
                        search_range = (0, self.FEATURE_NUM) if unit.owner else (self.FEATURE_NUM, self.FEATURE_NUM * 2)
                        if not self.is_free(j, search_range) and self.in_range(*self.get_raw_column(j), new_raw,
                                                                               new_col, diapason):
                            self.field_[moveRaw + 1 + i][j] = 1
        return

    def move_attack_matrix(self):
        def raw_creation(line):
            result = [0] * length
            raw = line // self.width_
            column = line - self.width_ * raw
            for i in range(length):
                raw_i = i // self.width_
                col_i = i - self.width_ * raw_i
                if not self.is_free(i, search_range):
                    if self.in_range(raw_i, col_i, raw, column, diapason):
                        result[i] = 1
            return result

        unit = self.units_[0]  # (Unit Name, Owner, raw, location)
        speed = cfg.Units[unit.name][5]
        diapason = cfg.Units[unit.name][6]
        length = self.height_ * self.width_
        location = unit.location
        raw, column = self.get_raw_column(location)
        search_range = (0, self.FEATURE_NUM) if unit.owner else (self.FEATURE_NUM, self.FEATURE_NUM * 2)

        return [raw_creation(i)
                if self.is_free(i) and self.can_move(*self.get_raw_column(i), raw, column, speed) else [0] * length
                for i in range(length)]

    def get_raw_column(self, location):
        raw = location // self.width_
        column = location - self.width_ * raw
        return raw, column

    def get_count(self, unit):
        """
        :param unit:(Unit name, Owner, Raw, Location)
        :return: count of unit at this location
        """
        return self.field_[unit.raw][unit.location] / cfg.Units[unit.name][3]

    def units_sort_(self):
        """ return list of tuples (Unit Name, Owner, raw, location)"""
        units_in_game = [Unit(cfg.index2name[i % self.FEATURE_NUM], i // self.FEATURE_NUM, i, j)
                         for i in range(self.FEATURE_NUM * 2) for j in range(len(self.field_[i])) if self.field_[i][j]]
        return sorted(units_in_game, key=lambda x: cfg.Units[x.name][-3], reverse=True), len(units_in_game)

    def check_winner(self, player):
        """
        Check if game is ended. Return 0 if there are no winner, 1 if first player win, and -1 otherwise
        """
        ifeatures = [(0, self.FEATURE_NUM), (self.FEATURE_NUM, self.FEATURE_NUM * 2)]
        player_win = [sum(map(sum, self.field_[start:end])) for start, end in ifeatures]
        return 0 if player_win[0] and player_win[1] else 1 if player_win[player] else -1

    def is_free(self, location, start_end=None):
        """
        Search any units on 'location'
        :param location: <int>, index, where should search
        :param start_end: <tuple(<int>, <int>)>, indexes of raws for search, default - all possible
        :return: True if no units is found, False otherwise
        """
        start_end = start_end if start_end else (0, self.FEATURE_NUM * 2)  # indexes for searching
        for i in range(*start_end):
            if self.field_[i][location] != 0:
                return False
        return True

    @staticmethod
    def can_move(raw_to, col_to, raw, column, speed):
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

    @staticmethod
    def calculate_damage(attacker, att_count, defender):
        """
        calculate damage that unit 'attacker' inflict to 'defender'
        :param attacker: <int> index of unit that attack
        :param att_count: <int> unit count in attack army
        :param defender: <int> index of unit that defence
        :return: <int> number of defeated life
        """
        attacker_name = cfg.index2name[attacker]
        defender_name = cfg.index2name[defender]
        damage = cfg.Units[attacker_name][2]  # get damage
        odds = cfg.Units[attacker_name][0] - cfg.Units[defender_name][1]  # difference between attack and defend
        damage_coeff = (1.0 + 0.1 * sign(odds)) ** abs(odds)  # coefficient of damage
        return int(damage * att_count * damage_coeff)

    @property
    def field(self):
        return self.field_

    @property
    def playerTurn(self):
        return self.units_[0].owner

    @property
    def id(self):
        return repr(self.FEATURE_NUM) + "&&" + repr(self.height_) + "&&" + repr(self.width_) + "&&" + \
               repr(self.units_) + "&&" + repr(self.field_)

    def from_id(self, identity):
        self.FEATURE_NUM, self.height_, self.width_, self.units_, self.field_ = map(eval, identity.split("&&"))
        self.fill_actions()
        return self

    def __eq__(self, other):
        if self.FEATURE_NUM != other.FEATURE_NUM or self.height_ != other.height_ or self.width_ != other.width_:
            return False
        if self.units_ != other.units_ or self.field_ != other.field_:
            return False
        return True

    def fight(self, attacker, defender):
        """
        Emulate fight between units 'attacker' and 'defender'
        :param attacker: (<int> raw, <int> location), number of unit who attack and his position in the field
        :param defender: (<int> raw, <int> location), number of unit who defends and his position in the field
        :return: (<int>: 'attacker' unit total life after attack, <int>: 'defender' unit total life after attack )
        """
        att_count = self.get_count(self.units_[0])
        # calculate total damage that unit 'attacker' inflict to 'defender'
        damage = self.calculate_damage(attacker[0], att_count, defender[0])
        damage = damage if damage > 1 else 1
        defender_life = self.field_[defender[0]][defender[1]] - damage  # total life of 'defender' unit
        defender_life = defender_life if defender_life > 0 else 0  # after 'attacker' attacked

        # calculate total damage that unit 'defender' inflict to 'attacker'
        attacker_life = self.field_[attacker[0]][attacker[1]]
        def_count = defender_life / cfg.Units[cfg.index2name[defender[0]]]
        if defender_life != 0:
            damage = self.calculate_damage(defender[0], def_count, attacker[0])
            damage = damage if damage > 1 else 1
            attacker_life = self.field_[attacker[0]][attacker[1]] - damage
            attacker_life = attacker_life if attacker_life > 0 else 0

        return attacker_life, defender_life

    def take_action(self, actions):
        """
        Perform action for current unit, and change game statement on new.
        Basic algorithm:
            1) Choose Maximum from movements
            2) If maximum value is more than 0:
                2.1) Check if this movement is allowed, return that game is loosed if it doesn't
                2.2) Make movement
            3) Choose maximum from attacks
            4) If maximum value is more than 0:
                4.1) Check possibilities of this attack, return that game is loosed if it doesn't
                4.2) Make attack
            5) Apply all changes on current state of game
            6) Check status for current gamer and return it
        :param actions: <list>, length = (self.height_ * self.width_) * 2; first half of list is associated with
                                                                            movements, second - with attacks
        :return: <int>, 0 if game is not ended, -1 if game is loosed or action is not allowed, 1 if game is wined
        """

        actions = [actions[:self.height_ * self.width_], actions[self.height_ * self.width_:]]
        movement = np.argmax(actions[0])
        current_unit = self.units_[0]  # (Name, Owner, Raw, Location)
        unit_raw = self.field_[current_unit.raw]
        player = current_unit.owner
        current_location = current_unit.location
        location_raw, location_col = self.get_raw_column(current_location)
        speed = cfg.Units[current_unit.name][5]
        distance = cfg.Units[current_unit.name][6]

        location_raw_new, location_col_new = self.get_raw_column(movement)

        # make move
        if actions[0][movement] > 0:
            # check that unit can move to this location
            if not self.can_move(location_raw_new, location_col_new, location_raw, location_col, speed):
                return Status.lose, None
            # check that location is free, except case when new and old locations are equal
            if current_unit.location != movement and not self.is_free(movement):
                return Status.lose, None

        # make attack
        defender_location = np.argmax(actions[1])
        if actions[1][defender_location] > 0:
            defender = self.find_unit(defender_location)
            if defender == -1:
                return Status.lose, None
            # check that there is unit in defenders location and it is enemy
            defender_owner = defender.owner
            def_raw, def_col = self.get_raw_column(defender_location)
            if defender_owner == -1 or defender_owner == player:
                return Status.lose, None
            # check that current unit able to attack defender unit
            if not self.in_range(def_raw, def_col, location_raw_new, location_col_new, distance):
                return Status.lose, None
            a_life, d_life = self.fight(current_unit.position, defender.position)
            if a_life < 0 or d_life < 0:
                raise Exception("fight method returned wrong values, att = {}, def = {}".format(a_life, d_life))

            # we need to remove attacker or defender units from queue if their total life is equal to 0
            if d_life == 0:
                self.change_unit(defender, GodHand.kill)
            if a_life == 0:
                self.change_unit(current_unit, GodHand.kill)
            unit_raw[current_location], self.field_[defender.raw][defender_location] = a_life, d_life

        # make movement if all checks are passed
        current_id = self.id
        unit_raw[current_location], unit_raw[movement] = unit_raw[movement], unit_raw[current_location]
        self.change_unit(Unit(name=current_unit.name, owner=current_unit.owner, raw=current_unit.raw,
                              location=movement), GodHand.move)
        self.fill_actions()
        winner = self.check_winner(player)
        result = self.id
        self.from_id(current_id)
        return winner, result

    def find_unit(self, location):
        units = list(filter(lambda x: x.location == location, self.units_))
        if len(units) > 1:
            raise Exception("Founded more then one units in one location")
        if len(units) == 0:
            return -1
        return units[0]

    def change_unit(self, unit, action):
        if action == GodHand.kill:
            self.units_.remove(unit)
        elif action == GodHand.move:
            # delete unit from start of queue and add 'unit' to the end
            self.units_.remove(self.units_[0])
            self.units_.append(unit)
        elif action == GodHand.add:
            self.units_.append(unit)


def generate_units_array(number, player_percent, weight, height):
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


# tmp = generate_units_array(8, 0.5, 15, 11)
tmp = [(4, 14, 'Rakshasa rani', 1, 27), (7, 12, 'Djinn sultan', 1, 57), (7, 9, 'Pit fiend', 1, 62),
       (9, 6, 'Master gremlin', 1, 39), (5, 11, 'Imperial griffin', 0, 73), (10, 8, 'Steel golem', 0, 55),
       (4, 1, 'Priest', 0, 45), (3, 1, 'Hell charger', 0, 19), (0, 0, 'Peasant', 1, 333)]

print(tmp)
game = Game(weight=15, height=11, units=tmp)
lol = game.id

from numpy.random import random
test_repetitions = 10000000
dimen = 15*11*2
length = 15*11
test_input = [random(dimen) for _ in range(test_repetitions)]

for test in test_input:
    tmp = game.take_action(test)
    if tmp != (Status.lose, None):
        print('*'*100)
        print("input = ", end=" ")
        move = np.argmax(test[:length])
        attack = np.argmax(test[length:])
        print(move, attack)
        print("output = ", end=" ")
        print(game.take_action(test))
        print("units = ", end=" ")
        print([str(i) for i in game.units_])


print([str(i) for i in game.units_])
