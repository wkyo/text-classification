# coding: utf-8
def uncamelize(s):
    """uncamelize string, such as `HelloWorld` -> `hello_world`"""
    array = []
    last_ch = None
    for ch in s:
        if ch.isupper():
            if last_ch and last_ch.islower():
                array.append('_')
            array.append(ch.lower())
        else:
            array.append(ch)
        last_ch = ch
    return ''.join(array)
