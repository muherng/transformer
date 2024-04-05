#file defines helper functions for constructing datasets
import numpy as np
import math
from random import randrange, choice
import itertools

#this function recursively creates 
#lhs then rhs according to pemdas
def gen_data(self,expr_type,params): 
    #many if statements to catch all cases
    if expr_type == 'arithmetic': 
        num_list,op_list = params[expr_type]
        if np.where(num_list < 0)[1].size != 0 and np.where(op_list==self.minus_token)[1].size != 0: 
            raise ValueError('Negative numbers only appear in initialization, operations must all be add')
        
        flag = -1
        while np.shape(num_list)[1] > 1:
            #creates data lhs and rhs and new num_list,op_list
            #print('nums: ', num_list[0,:])
            #print('ops: ', op_list[0,:])
            data_new,num_list,op_list = self.gen_equation(num_list,op_list)
            #print('equation: ', data_new[0,:])
            if flag == -1:
                data = data_new
                flag += 1
            else: 
                data = np.concatenate([data, data_new], axis=0)
    else: 
        raise NotImplementedError
    #raise ValueError('STOP')
    data = data.astype(int)
    return data

def gen_equation(self,num_list,op_list):
    num_list_rhs, op_list_rhs = self.one_step(num_list, op_list)
    
    #number of digits + 1 negative token + 1 operator token * number of arguments * 2 for both lhs and rhs 
    #plus start, equals, and eos tokens
    #note that length depends on num_args instead of the cols in num_list
    length =  2*(self.number_length + 2)*self.num_args + 3 
    row,col = np.shape(num_list)
    data = np.zeros((row,length))
    for i in range(row):
        num = num_list[i,:]
        op = op_list[i,:]
        num_rhs = num_list_rhs[i,:]
        op_rhs = op_list_rhs[i,:]
        equation = np.concatenate([[self.start_token], self.gen_line(num, op),
                                   [self.end_token], self.gen_line(num_rhs, op_rhs), [self.eos_token]]) 
        pad = np.array([self.padding_token]*(length - len(equation)))
        data[i,:] = np.concatenate([equation,pad])
    return data, num_list_rhs, op_list_rhs
    
#generates a single line from numbers and operations without start and equal tokens
def gen_line(self, num, op):
    out = []
    for i in range(len(num)):
        out.extend(self.num_to_digits(num[i]))
        if i < len(op):
            out.append(op[i])
    #change +- to -
    remove = []
    for i in range(len(out)-1):
        if out[i] == self.add_token and out[i+1] == self.minus_token:
            remove.append(i)
    out = [elem for it,elem in enumerate(out) if it not in remove]
    return out 
        
    

#function for evaluating the first pemdas operation in num_list and op_list 
def one_step(self,num_list,op_list):
    
    #if there's a multiplication, evaluate it
    row_op,col_op = np.shape(op_list)
    row, col_num = np.shape(num_list)
    if col_op != col_num - 1:
        raise ValueError('there must be one fewer operation than numbers')
    if row_op != row:
        raise ValueError('op_list and num_list must have same number of rows')
    
    #new op list will have one fewer operation 
    op_list_new = np.zeros((row,col_op-1))
    num_list_new = np.zeros((row, col_num-1))
    for i in range(row): 
        op_row = op_list[i,:]
        num_row = num_list[i,:]
        if np.where(op_row == self.mult_token)[0].size > 0:
            first_mult = np.where(op_row == self.mult_token)[0][0]
            num_list_new[i,:first_mult] = num_row[:first_mult]
            num_list_new[i,first_mult] = num_row[first_mult]*num_row[first_mult+1]
            #Note: numpy arrays don't index out of bounds for slicing--weird
            num_list_new[i,first_mult+1:] = num_row[first_mult+2:]
            #update operation list
            op_row = list(op_row)
            op_row.pop(first_mult)
            op_list_new[i,:] = op_row
        else: 
            #no multiplication, so peform first add
            first_add = np.where(op_row == self.add_token)[0][0]
            num_list_new[i,:first_add] = num_row[:first_add]
            num_list_new[i,first_add] = num_row[first_add] + num_row[first_add+1]
            num_list_new[i,first_add+1:] = num_row[first_add+2:]
            op_row = list(op_row)
            op_row.pop(first_add)
            op_list_new[i,:] = op_row
    
    #TODO: implement add and minus 
    return (num_list_new, op_list_new)

#takes a string and return a list of ints 
def num_to_digits(self,number):
    out = []
    for num in list(str(int(number))):
        if num == '-':
            out.append(self.minus_token)
        else: 
            out.append(int(num))
    return out
        

def gen_lhs(self,num_list,op_list):
    #length depends on operations, we can make it longer than it ought to be 
    #number of digits + 1 operator token * number of arguments * 2 for both lhs and rhs 
    #plus start, equals, and eos tokens
    length =  2*(self.number_length + 1)*self.num_args + 3 
    #obtain longest number of digits in num_list
    row,col = np.shape(num_list)
    data = np.zeros((row,length))
    for i in range(row):
        numbers = num_list[i,:]
        row = [self.start_token]
        for digit in range(len(numbers)):
            row = row + self.num_to_digits(numbers[digit]) + [self.add_token]
        #remove final add token
        row = row[:-1]
        #append "=" token
        row = row + [self.end_token]
        row = row + (length - len(row))*[self.padding_token]
        data[i,:] = row
    return data

def gen_rhs(self,num_list, op_list):
    raise NotImplementedError
    
def update(self,num_list,op_list):
    raise NotImplementedError

def combine(self,lhs,rhs):
    raise NotImplementedError 

def innerprod_lhs(self,num_list, bs, data=None):
    #first add start token 
    if data is not None:
        #fill with -1's in the shape of data
        data_new = np.full(np.shape(data),-1)
        data_new[:,0] = np.full(bs, self.start_token)
        for i in range(np.shape(data)[0]):
            row = data[i,:]
            #TODO: these raise errors are all wrong, np.where returns tuple 
            #change to np.where()[0]
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
            row = row + self.num_to_digits(numbers[digit]) + [self.mult_token] + self.num_to_digits(numbers[digit+1])
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
                rhs = rhs + self.num_to_digits(numbers[digit]*numbers[digit+1])
            else: 
                rhs = rhs + self.num_to_digits(numbers[digit]) + [self.mult_token] + self.num_to_digits(numbers[digit+1])
                
            if digit < self.num_args-2:
                rhs = rhs + [self.add_token]
        rhs = rhs + [self.eos_token]
        row[start:start+len(rhs)] = rhs
        row[start+len(rhs):] = (len(row) - start - len(rhs))*[self.padding_token]
    return data

#Only create data for sums that exist in the dataset? 
#It should be general summing.  or both? 
#for now general summing up to two digits
def sum_data(self,num_list):
    bs,num_args = np.shape(num_list)
    if num_args == 1:
        raise ValueError('num_args must be bigger than 1')
    bs_mini = int(bs/(num_args-1))
    #bs_mini = bs
    length =  2*(self.number_length + 1)*self.num_args + 3 
    for outer_loop in range(self.num_args - 1): 
        data_new = np.full((bs_mini,length),-1)
        for i in range(bs_mini):
            numbers = num_list[i,:]
            row = [self.start_token]
            for digit in range(len(numbers)):
                row = row + self.num_to_digits(numbers[digit]) + [self.add_token]
            #remove final add token
            row = row[:-1]
            #append "=" token
            row = row + [self.end_token]
            #add first two numbers 
            row = row + self.num_to_digits(numbers[0] + numbers[1]) + [self.add_token]
            for digit in range(2,len(numbers)):
                row = row + self.num_to_digits(numbers[digit]) + [self.add_token]
            row = row[:-1]
            row = row + [self.eos_token]
            row = row + (length - len(row))*[self.padding_token]
            data_new[i,:] = row
        if outer_loop == 0: 
            data = data_new 
        else: 
            data = np.concatenate([data,data_new],axis=0)
        #for all but the last iteration update num_list
        if outer_loop < self.num_args-2: 
            num_list_new = np.zeros((np.shape(num_list)[0], np.shape(num_list)[1] - 1))
            num_list_new[:,0] = num_list[:,0] + num_list[:,1]
            if np.shape(num_list)[1] >= 2: 
                num_list_new[:,1:] = num_list[:,2:]
            num_list = num_list_new
            #cast to int 
            num_list = num_list.astype(int)
    return data 

def pretty_print(self,data):
    token_dict = {self.start_token:"", self.add_token:"+", 
                  self.minus_token:"-", self.mult_token:"*", 
                  self.end_token:"=", self.eos_token:"", self.padding_token:""}
    for row in data: 
        row_obj = row.astype('object')
        for token in [self.start_token, self.add_token, self.minus_token, self.mult_token, self.end_token, self.eos_token, self.padding_token]:
            # Replace every occurrence of a specified integer with 'foo'
            row_obj[row_obj == token] = token_dict[token]
            row_obj = row_obj[row_obj != '']
        print(np.array2string(row_obj,formatter={'all': lambda x: str(x)})) 
