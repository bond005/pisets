import math


def time_to_str(time_val: float) -> str:
    if not isinstance(time_val, float):
        raise ValueError(f'The time value {time_val} is not floating-point!')
    if time_val < 0.0:
        raise ValueError(f'The time value {time_val} is negative!')
    hours = int(math.floor(time_val / 3600.0))
    if hours == 0:
        s = '00'
    elif hours < 10:
        s = '0' + str(hours)
    else:
        s = str(hours)
    s += ':'
    time_val -= hours * 3600.0
    minutes = int(math.floor(time_val / 60.0))
    if minutes < 10:
        s += '0' + str(minutes)
    else:
        s += str(minutes)
    time_val -= minutes * 60.0
    s += ':'
    s += ('{0:06.3f}'.format(time_val)).replace('.', ',')
    return s
