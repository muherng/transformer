#file defines helper functions for constructing datasets
import numpy as np
import math
from random import randrange, choice
import itertools

#takes a string and return a list of ints 
def to_digits(number):
    return [int(num) for num in list(str(number))]

def innerprod_lhs(self,num_list, bs, data=None):
    #first add start token 

    if data is not None:
        #fill with -1's in the shape of data
        data_new = np.full(np.shape(data),-1)
        data_new[:,0] = np.full(bs, self.start_token)
        for i in range(np.shape(data)[0]):
            row = data[i,:]
            if len(np.where(row == self.end_token)) > 1:
                raise ValueError('More than one equal sign in row')
            if len(np.where(row==self.eos_token)) > 1:
                raise ValueError('More than one eos token in row')
            end_token = np.where(row == self.end_token)[0][0]
            eos_token = np.where(row == self.eos_token)[0][0]
            #data after equals not including eos_token
            row_rhs = row[end_token+1:eos_token]  
            data_new[i,1:len(row_rhs)+1] = row_rhs
            data_new[i,len(row_rhs)+1] = self.end_token
        return data_new 
            
    #each number is followed by a symbol * or +, 
    #there are two copies on left and right of equal sign
    #start, equal, and eos are three more 
    length = 2*(self.number_length + 1)*self.num_args + 3 
    data = np.full((bs,length),-1)
    #next add every pair of digits with * and then +
    for i in range(bs):
        numbers = num_list[:,i]
        row = [self.start_token]
        for digit in range(0,self.num_args,2):
            #print('to_digits: ', str_to_int(numbers[digit]))
            row = row + to_digits(numbers[digit]) + [self.mult_token] + to_digits(numbers[digit+1])
            if digit < self.num_args - 2:
                row = row + [self.add_token]
        row = row + [self.end_token]
        #print('row: ', row)
        data[i,:len(row)] = row
        #print('data[i,:]', data[i,:])
    return data

def innerprod_rhs(self,num_list,bs,data,pointer=0,lhs=False): 
    if pointer%2 == 1:
        raise ValueError('can only multiply first 2,4,6,8... digits')
    #add output sum of first two digits
    for i in range(bs):
        row = data[i,:]
        start = np.where(row == -1)[0][0]
        #print('start: ', start)
        numbers = num_list[:,i]
        rhs = []
        for digit in range(0,self.num_args,2):
            if digit < pointer: 
                rhs = rhs + to_digits(numbers[digit]*numbers[digit+1])
            else: 
                rhs = rhs + to_digits(numbers[digit]) + [self.mult_token] + to_digits(numbers[digit+1])
                
            if digit < self.num_args-2:
                rhs = rhs + [self.add_token]
        rhs = rhs + [self.eos_token]
        row[start:start+len(rhs)] = rhs
        row[start+len(rhs):] = (len(row) - start - len(rhs))*[self.padding_token]
    return data
