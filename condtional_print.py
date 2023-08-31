# -*- coding:utf-8 -*-
"""
Author:Alarak
Date:2023/07/20

put some visual in console when printing losses
"""


def conditional_print(losses, loss, name):
    print(name, end='')
    if losses == []:
        print('\033[4;32m' + f'{loss:.2e}' + '\033[0m', end=' ')
    else:
        if loss < min(losses):
            print('\033[4;32m' + f'{loss:.2e}' + '\033[0m',
                  end=' ')  # if loss now is the smallest in history, print in green with underline
        elif loss < losses[-1]:
            print('\033[32m' + f'{loss:.2e}' + '\033[0m',
                  end=' ')  # if loss now is smaller than last one, print in green
        else:
            print('\033[31m' + f'{loss:.2e}' + '\033[0m', end=' ')  # otherwise, print in red
