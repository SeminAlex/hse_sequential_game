from enum import Enum
from copy import deepcopy as cp
from random import randrange, sample
import config as cfg


def _create_zero_line(length):
    return [0 for _ in range(length)]


class Status(Enum):
    win = 1
    lose = -1
    draw = 0
    not_ended = -2


class Game:
    __slots__ = ["field_", "player_", "units_", "height_", "weight_", "FEATURE_NUM", "status", ]
    """
    field_ - matrix: FEATURE_NUM x (height*weight), where each row has length= height*weight
             field_[0-FEATURE_NUM] - binary arrays, 1 in raw i means, that unit, which name is unit_names[i] is located 
                                     in this cell and it is all units of FIRST player
             field_[FEATURE_NUM-FEATURE_NUM*2] - binary arrays, same as previous but for SECOND player
             field_[FEATURE_NUM*2+1] -  
                          
    
    player_       - binary variable that indicate which turn is now
    height_       - int variable, shows height of field
    weight_       - int variable, shows weight of field
    FEATURE_NUM   - int variable, contain number of rows in field_ matrix
    status        - Status variable, can be any value from 'Status' class
    """

    def __init__(self, height, weight, units=None, field=None):
        self.height_ = height
        self.weight_ = weight
        self.FEATURE_NUM = len(cfg.units_names)
        self.status = Status.not_ended
        if field:
            self.field_ = cp(field)
        else:
            self.__generate_field(units)

    def __generate_field(self, units=None):
        length = self.weight_ * self.height_
        self.field_ = [_create_zero_line(length) for _ in range(self.FEATURE_NUM)]

        if units:
            if len(units) < 1 or len(units[0]) != 4:
                raise Exception("ERROR: List of units must contain at least one tuple like (X, Y, UNIT_NAME, PLAYER)")
            for x, y, unit_name, player in units:
                attributes = cfg.Units.get(unit_name, None)
                if not attributes:
                    raise Exception("ERROR: Unit with name {} doesn't exist. Possible unit names : {}"
                                    .format(unit_name, Units.keys()))
                start = 1 + len(attributes) * player
                coordinate = x * self.weight_ + y
                self.field_[0][coordinate] = 1
                for index in range(len(attributes)):
                    self.field_[start + index][coordinate] = attributes[index]
        pass

    def fight(self, attacker, defender):
        """
        Emulate fight between units 'attacker' and 'defender'
        :param attacker: int, position of unit who attack
        :param defender: int, position of unit who defends
        :return: False if attacker or defender are exist, True - otherwise
        """

        pass

    def is_unit_exist(self, unit):
        if self.field_[0][unit]:
            # TODO: think about deleting of hardcodes
            if self.field_[1][unit] or self.field_[15][unit]:
                return True
        return False


def genereate_units_array(number, player_percent, weight, height):
    def indext2coordinate(index):
        w = index // weight
        h = index - weight * w
        return w, h

    coord_inexes = sample(range(weight * height), number)
    coordinates = [indext2coordinate(i) for i in coord_inexes]
    ownership = [1 if i < number * player_percent else 0 for i in range(number)]
    units = [(*coordinates[i], *sample(Units.keys(), 1), ownership[i]) for i in range(number)]
    return units


game = Game(15, 11)

# tmp = genereate_units_array(8, 0.5, 15, 11)
tmp = [(2, 3, 'slender_men', 1), (2, 13, 'slender_men', 1), (9, 13, 'slender_men', 1), (7, 9, 'archer', 1),
       (9, 9, 'archer', 0), (1, 3, 'slender_men', 0), (1, 4, 'archer', 0), (2, 8, 'archer', 0)]
print(tmp)
game = Game(weight=15, height=11, units=tmp)
print(game)
