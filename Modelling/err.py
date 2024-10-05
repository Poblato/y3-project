from enum import IntEnum

class Err(IntEnum):
    OK = 0
    NonFatalRangeError = -2
    GenericNonFatalError = -1
    GenericFatalError = 1
    RangeError = 2
    ValueError = 3
    NullError = 4

    def ShowError(err):
        errorCodes = {
            0: "OK",
            -2: "Non-Fatal Range Error",
            -1: "Generic Non-Fatal Error",
            1: "Generic Fatal Error",
            2: "Range Error",
            3: "Value Error",
            4: "Null Error"
        }
        return errorCodes[err]