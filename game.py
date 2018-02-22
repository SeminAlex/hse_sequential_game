from enum import Enum
from copy import deepcopy as cp
from random import randrange, sample
from config import Units

game_matrix = list()  # matrix asossiated with game field

N = 10
M = 12


def _create_zero_line(length):
    return [0 for _ in range(length)]


class Status(Enum):
    win = 1
    lose = -1
    draw = 0
    isnot_ended = -2


class Game:
    __slots__ = ["field_", "player_", "units_", "height_", "weight_", "FEATURE_NUM", "status", ]
    """
    field_ - matrix: FEATURE_NUM x (height*weight), where each row has length= height*weight
             field_[0] - binary array, 1 means there is unit in this location, and 0 - otherwise
             field_[1-10] - arrays associated with unit features for FIRST player : 
                                 1) attack, 2) defence, 3) damage min, 4) damage max, 5) health, 6) initiative, 
                                 7) speed, 8) shoots(??), 9) spellpoints(??), 10) b-pop(???)
             field_[11-20] - arrays associated with unit features for SECOND player :
                                 11) attack, 12) defence, 13) damage min, 14) damage max, 15) health, 16) initiative,
                                 17) speed, 18) 1shoots(??), 19) spellpoints(??), 20) b-pop(???)
             field_[21] - binary array with range, where current unit can move(1 means unit can move in location, 
                            0 - otherwise) 
             field_[22] - binary array with location which current unit can attack ONLY if he made movement in this 
                           round: 1 means that location can be attacked by unit, 0 - otherwise
             field_[23] - binary array with locations that can be attacked if unit didn't made movements in this round:
                            1 means location can be attacked, 0 - otherwise
             field_[24] - 
             
    
    player_       - binary variable that indicate which turn is now
    height_       - int variable, shows height of field
    weight_       - int variable, shows weight of field
    FEATURE_NUM   - int variable, contain number of rows in field_ matrix
    status        - Status variable, can be any value from 'Status' class
    """

    def __init__(self, height=15, weight=11, units=None, field=None):
        self.height_ = height
        self.weight_ = weight
        self.FEATURE_NUM = 24
        self.status = Status.isnot_ended
        if field:
            self.field_ = cp(field)
        else:
            self.__generate_field(units)

    def __generate_field(self, units=None):
        length = self.weight_ * self.height_
        self.field_ = [_create_zero_line(length) for _ in range(self.FEATURE_NUM)]

        if units:
            if len(units) < 1 or len(units[0]) != 2:
                raise Exception("ERROR: List of units must contain at least one tuple like (X, Y, UNIT_NAME, PLAYER)")
            for unit_name, player in units:
                attributes = Units.get(unit_name, default=None)
                if not attributes:
                    raise Exception("ERROR: Unit with name {} doesn't exist. Possible unit names : {}"
                                    .format(unit_name, Units.keys()))
                start = 1 + len(attributes) * player
                for index in range(len(attributes)):
                    self.field_[start + index] = attributes[index]
        pass


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


game = Game()

tmp = genereate_units_array(8, 0.5, 15, 11)
print(tmp)
game = Game(weight=15, height=11, units=tmp)
