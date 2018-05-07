from enum import Enum
from random import randrange, sample

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
    next = 2


class Unit:
    __slots__ = ["name", "owner", "raw", "location", "life"]

    def __init__(self, name, owner, raw, location, life):
        self.name = name
        self.owner = owner
        self.raw = raw
        self.location = location
        self.life = life

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
        return str(list(zip(self.__slots__, [self.name, self.owner, self.raw, self.location, self.life])))

    def __repr__(self):
        return "Unit(name={}, owner={},raw={},location={}, life={})".format(
            repr(self.name), self.owner, self.raw, self.location, self.life)

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone


class Game:
    __slots__ = ["player_", "units_", "height_", "width_", "FEATURE_NUM", ]
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
    units_        - Unit object array which determinate queue of units turns
    """

    def __init__(self, height, weight, units=None, from_id=False):
        self.height_ = height
        self.width_ = weight
        self.FEATURE_NUM = len(cfg.name2index)
        if not from_id:
            self.__generate_units(units)
        else:
            self.units_ = units

    def __generate_units(self, units=None):
        self.units_ = list()
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
                self.units_.append(Unit(name=unit_name,
                                        owner=player,
                                        raw=raw,
                                        location=coordinate,
                                        life=cfg.Units[unit_name][3] * count))
            self.units_ = self.units_sort_()
        return

    def move_attack_matrix(self):
        def raw_creation(line):
            result = [0] * length
            raw_tmp = line // self.width_
            column_tmp = line - self.width_ * raw_tmp
            for i in range(length):
                raw_i = i // self.width_
                col_i = i - self.width_ * raw_i
                if not self.is_free(i, search_range):
                    if self.in_range(raw_i, col_i, raw_tmp, column_tmp, diapason):
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

    @staticmethod
    def get_count(unit):
        """
        :param unit:(Unit name, Owner, Raw, Location)
        :return: count of unit at this location
        """
        return unit.life / cfg.Units[unit.name][3]

    def units_sort_(self):
        """ return list of tuples (Unit Name, Owner, raw, location, life)"""
        return sorted(self.units_, key=lambda x: cfg.Units[x.name][-3], reverse=True)

    def check_winner(self, player):
        """
        Check if game is ended. Return 0 if there are no winner, 1 if first player win, and -1 otherwise
        """
        player_win = [len(list(filter(lambda x: True if x.owner == 0 else False, self.units_))),
                      len(list(filter(lambda x: True if x.owner == 1 else False, self.units_)))]
        return 0 if player_win[0] and player_win[1] else 1 if player_win[player] else -1

    def is_free(self, location, start_end=None):
        """
        Search any units on 'location'
        :param location: <int>, index, where should search
        :param start_end: <tuple(<int>, <int>)>, indexes of raws for search, default - all possible
        :return: True if no units is found, False otherwise
        """
        start, end = start_end if start_end else (0, self.FEATURE_NUM * 2)  # indexes for searching
        for unit in self.units_:
            if start <= unit.raw < end and unit.location == location:
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

    def calculate_damage(self, attacker, att_count, defender):
        """
        calculate damage that unit 'attacker' inflict to 'defender'
        :param attacker: <int> index of unit that attack
        :param att_count: <int> unit count in attack army
        :param defender: <int> index of unit that defence
        :return: <int> number of defeated life
        """
        attacker_name = cfg.index2name[attacker % self.FEATURE_NUM]
        defender_name = cfg.index2name[defender % self.FEATURE_NUM]
        damage = cfg.Units[attacker_name][2]  # get damage
        odds = cfg.Units[attacker_name][0] - cfg.Units[defender_name][1]  # difference between attack and defend
        damage_coeff = (1.0 + 0.1 * sign(odds)) ** abs(odds)  # coefficient of damage
        return int(damage * att_count * damage_coeff)

    @property
    def field(self):
        length = self.height_ * self.width_
        field = [_create_zero_line(length) for _ in range(self.FEATURE_NUM * 2)]
        field.extend(self.move_attack_matrix())
        return field

    @property
    def player_turn(self):
        return self.units_[0].owner

    @property
    def id(self):
        return repr(self.FEATURE_NUM) + "&&" + repr(self.height_) + "&&" + repr(self.width_) + "&&[" + \
               ','.join(map(repr, self.units_)) + "]&&"

    @staticmethod
    def from_id(identity):
        _, height, width, units = map(eval, identity.split("&&"))
        return Game(height, width, units=units, from_id=True)

    def __eq__(self, other):
        if self.FEATURE_NUM != other.FEATURE_NUM or self.height_ != other.height_ or self.width_ != other.width_:
            return False
        if self.units_ != other.units_:
            return False
        return True

    def fight(self, attacker, defender):
        """
        Emulate fight between units 'attacker' and 'defender'
        :param attacker: <Unit>, number of unit who attack and his position in the field
        :param defender: <Unit>, number of unit who defends and his position in the field
        :return: (<int>: 'attacker' unit total life after attack, <int>: 'defender' unit total life after attack )
        """
        att_count = self.get_count(attacker)
        # calculate total damage that unit 'attacker' inflict to 'defender'
        damage = self.calculate_damage(attacker.raw, att_count, defender.raw)
        damage = damage if damage > 1 else 1
        defender_life = defender.life - damage  # total life of 'defender' unit
        defender_life = defender_life if defender_life > 0 else 0  # after 'attacker' attacked

        # calculate total damage that unit 'defender' inflict to 'attacker'
        attacker_life = attacker.life
        def_count = defender_life / cfg.Units[defender.name][3]
        if defender_life != 0:
            damage = self.calculate_damage(defender.raw, def_count, attacker.raw)
            damage = damage if damage > 1 else 1
            attacker_life -= damage
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
        result_units = self.units_[:]
        current_unit = result_units[0]
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
        a_life, d_life, defender = None, None, None
        defender_location = np.argmax(actions[1])
        if defender_location == current_unit.location:
            return Status.lose, None
        if actions[1][defender_location] > 0:
            defender = self.find_unit(result_units, defender_location)
            if not defender:
                return Status.lose, None
            # check that there is unit in defenders location and it is enemy
            defender_owner = defender.owner
            def_raw, def_col = self.get_raw_column(defender_location)
            if defender_owner == -1 or defender_owner == player:
                return Status.lose, None
            # check that current unit able to attack defender unit
            if not self.in_range(def_raw, def_col, location_raw_new, location_col_new, distance):
                return Status.lose, None
            a_life, d_life = self.fight(current_unit, defender)
            if a_life < 0 or d_life < 0:
                raise Exception("fight method returned wrong values, att = {}, def = {}".format(a_life, d_life))

        # make movement if all checks are passed
        self.change_units(result_units, current_unit, GodHand.move, location=movement)

        # we need to remove attacker or defender units from queue if their total life is equal to 0
        if a_life and d_life and defender:
            if d_life == 0:
                self.change_units(result_units, defender, GodHand.kill)
            if a_life == 0:
                self.change_units(result_units, self.units_[0], GodHand.kill)
            # make attack if all checks are passed
            current_unit.life = a_life
            defender.life = d_life
        self.change_units(result_units, defender, GodHand.next)
        winner = self.check_winner(player)
        return winner, Game(self.height_, self.width_, result_units, from_id=True)

    @staticmethod
    def find_unit(units, location):
        units = list(filter(lambda x: x.location == location, units))
        if len(units) > 1:
            raise Exception("Founded more then one units in one location")
        if len(units) == 0:
            return None
        return units[0]

    @staticmethod
    def change_units(units_list, unit, action, **kwargs):
        if action == GodHand.kill:
            units_list.remove(unit)
        elif action == GodHand.next:
            # delete unit from start of queue and add him to the end
            units_list.remove(units_list[0])
            units_list.append(unit)
        elif action == GodHand.add:
            units_list.append(unit)
        elif action == GodHand.move:
            new_location = kwargs["location"]
            index = units_list.index(unit)
            units_list[index].location = new_location


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


def main():
    # tmp = generate_units_array(8, 0.5, 15, 11)
    tmp = [(4, 14, 'Rakshasa rani', 1, 27), (7, 12, 'Djinn sultan', 1, 57), (7, 9, 'Pit fiend', 1, 62),
           (9, 6, 'Master gremlin', 1, 39), (5, 11, 'Imperial griffin', 0, 73), (10, 8, 'Steel golem', 0, 55),
           (4, 1, 'Priest', 0, 45), (3, 1, 'Hell charger', 0, 19), (0, 0, 'Peasant', 1, 333)]

    game = Game(weight=15, height=11, units=tmp)
    log = open("logger.log", "w")

    def test_one_unit(units, length):
        locations = [u[-1] for u in units]
        for i in locations:
            for movement in range(length):
                for attack in range(length):
                    if (movement not in locations and attack in locations) or movement == i:
                        template = "*********input (move: {}, attack: {}), result: {})"
                    else:
                        template = "input (move: {}, attack: {}), result: {})"
                    inputs = [0] * (length * 2)
                    inputs[movement] = 1
                    inputs[attack + length] = 1
                    result = game.take_action(inputs)
                    print(template.format(str(movement), str(attack), str(result)), file=log)

    def test_all(units):
        length = 15 * 11
        for _ in range(len(units)):
            print("*" * 75, end="", file=log)
            print("  " + str(units[0]) + "  ", end="", file=log)
            print("*" * 75, end="\n", file=log)
            test_one_unit(units, length)
            units.append(units[0])
            units.remove(units[0])

    for j in range(5):
        print(tmp)
        print("*" * 150, end="\n", file=log)
        st = str(tmp)
        seps_len = int(((150 - len(st)) / 2) - 4)
        print("*" * seps_len, end="  ", file=log)
        print(st, end="  ", file=log)
        print("*" * seps_len, end="\n", file=log)
        print("*" * 150, end="\n", file=log)
        test_all(tmp)
        tmp = generate_units_array(15 + j * 2, 0.5, 15, 11)

    log.close()


if __name__ == "__main__":
    main()
