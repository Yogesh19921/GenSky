import math
from random import randint


class vm:
    # all variables

    def getFitness(self):
        return self.fitness

    def __init__(self, target):

        self.general_purpose_register = []
        self.mode = []
        self.target = []
        self.input_register = []
        self.operator = []
        self.source_ip = []
        self.no_of_reg = 0
        self.fitness = 0
        self.classes = []
        self.no_of_instructions = 0
        self.multiFitness = []
        self.dominance_count = 0
        self.dominance_rank = 0
        self.defalut_registers = []

        self.false_positive=0
        self.true_positive = 0
        self.false_negative = 0
        self.true_negative = 0

        self.class_mapping(target)
        self.no_of_instructions = randint(len(self.classes) * 2, 20)
        self.reset()

        self.init_mode()
        self.init_target()
        self.init_operator()
        self.init_source_ip()
        self.init_multiFitness()

    def __lt__(self, other):
        try:
            a = 0

            flag = 0

            for i in range(0, len(self.multiFitness)):
                if (self.multiFitness[i] < other.multiFitness[i]):
                    flag = 1
                    break
                else:
                    if (self.multiFitness[i] > other.multiFitness[i]):
                        flag = 2
                        break

            if (flag == 1):
                for i in range(0, len(self.multiFitness)):
                    if (self.multiFitness[i] <= other.multiFitness[i]):
                        a = -1
                    else:
                        a = 0
                        break

            if (flag == 2):
                for i in range(0, len(self.multiFitness)):
                    if (self.multiFitness[i] >= other.multiFitness[i]):
                        a = 1
                    else:
                        a = 0
                        break

            if (a == 0):
                if (other.multiFitness[0] >= 70 and other.multiFitness[1] >= 50):
                    a = -1
                    self.dominance_rank = self.dominance_rank + 1
                    other.dominance_count= other.dominance_count + 1
                    return a

                if (self.multiFitness[0] >= 70 and self.multiFitness[1] >= 50):
                    a = 1
                    other.dominance_rank = other.dominance_rank + 1
                    self.dominance_count = self.dominance_count + 1
                    return a

                if (other.multiFitness[0] >= 60 and other.multiFitness[1] >= 40):
                    a = -1
                    self.dominance_rank = self.dominance_rank + 1
                    other.dominance_count = other.dominance_count + 1
                    return a

                if (self.multiFitness[0] >= 60 and self.multiFitness[1] >= 40):
                    a = 1
                    other.dominance_rank = other.dominance_rank + 1
                    self.dominance_count = self.dominance_count + 1
                    return a

            ''''''

        except Exception as e:
            print e.message

        return a

    def init_multiFitness(self):
        self.multiFitness = []
        for i in range(0, len(self.classes)):
            self.multiFitness.append(0)

    def class_mapping(self, target):
        # global classes
        self.classes = []
        m = 0
        for f in target:
            if (m < len(target)):
                if (f not in self.classes):
                    self.classes.append(f)
                    # self.multiFitness.append(0)
            m = m + 1

    # initializing GPR
    def init_general_purpose_register(self):
        # global general_purpose_register
        self.general_purpose_register = []
        self.defalut_registers = []
        c = 1.0
        for i in range(0, len(self.classes)):
            n = randint(0, 500)
            self.defalut_registers.append('R' + str(i))
            self.general_purpose_register.append(c)
            # c = c + 0.1

    # Initializing Random numbers to mode
    # 0 -> input, 1 -> register
    def init_mode(self):
        # global mode
        self.mode = []

        for i in range(0, self.no_of_instructions):
            self.mode.append(randint(0, 1))

    # Initializing target randomly
    def init_target(self):
        # global target
        self.target = []

        for i in range(0, self.no_of_instructions):
            self.target.append(randint(0, len(self.classes) - 1))

    # Initializing Operators randomly
    def init_operator(self):
        # global operator
        self.operator = []

        for i in range(0, self.no_of_instructions):
            self.operator.append(randint(0, 6))

    # Initializing source/Input parameters randomly
    def init_source_ip(self):
        self.source_ip = []

        for i in range(0, self.no_of_instructions):
            self.source_ip.append(randint(0, 255))

    # Reset the vm for next run
    def reset(self):
        self.init_general_purpose_register()

    # Mapping of numbers to operators
    def map_operator(self, op):
        if (op == 1):
            return '+'
        if (op == 2):
            return '-'
        if (op == 3):
            return '/'
        if (op == 4):
            return '*'

        else:
            return 'invalid'

    def fetch_decode_execute(self, input, target):
        # fetching instructions [0,0,1,45]

        if input is None:
            return

        self.reset()

        for i in range(0, self.no_of_instructions):
            _mode = self.mode[i]
            _target = self.target[i]
            _operator = self.operator[i]
            _source_ip = self.source_ip[i]

            if (_mode == 0):
                _register = (_source_ip % (len(input) - 1))
                # print i
                if (_operator == 0):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                        self.target[i]] + float(input[_register])
                if (_operator == 1):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                        self.target[i]] - float(input[_register])
                if (_operator == 2):
                    try:
                        self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                            self.target[i]] / float(input[_register])
                    except ZeroDivisionError:
                        print 'Zero Division not allowed'
                if (_operator == 3):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                        self.target[i]] * float(input[_register])

                if (_operator == 4):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                        self.target[i]] ** (2)
                if (_operator == 5):
                    self.general_purpose_register[self.target[i]] = math.sin(float(input[_register]))

                if (_operator == 6):
                    self.general_purpose_register[self.target[i]] = math.cos(float(input[_register]))



            else:
                _register = (_source_ip % len(self.classes))
                if (_operator == 0):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[self.target[i]] + \
                                                                    self.general_purpose_register[_register]
                if (_operator == 1):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[self.target[i]] - \
                                                                    self.general_purpose_register[_register]
                if (_operator == 2):
                    try:
                        self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                            self.target[i]] / float(input[_register])
                    except ZeroDivisionError:
                        print 'Zero Division not allowed'
                if (_operator == 3):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[self.target[i]] * \
                                                                    self.general_purpose_register[_register]
                if (_operator == 4):
                    self.general_purpose_register[self.target[i]] = self.general_purpose_register[
                                                                        self.target[i]] ** (2)

                if (_operator == 5):
                    self.general_purpose_register[self.target[i]] = math.sin(self.general_purpose_register[_register])

                if (_operator == 6):
                    self.general_purpose_register[self.target[i]] = math.cos(self.general_purpose_register[_register])

        max = 'R0'
        maxval = self.general_purpose_register[0]
        for i in range(0, len(self.classes)):
            if (maxval < self.general_purpose_register[i]):
                maxval = self.general_purpose_register[i]
                max = self.defalut_registers[i]

        return max
