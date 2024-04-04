import numpy as np
import math
from random import randrange, choice
import itertools
import torch
from data_constructor import *
#innerprod_lhs, innerprod_rhs, sum_data, pretty_print, gen_data

# Adding extra padding is an easy way to improve performance, as it gives the
# model more space to think. For example, without padding, the standard model
# kind=transformer-lstm gets accuracies (1, 0.95, 0.66), but we just five extra
# paddings on the left, it gets (1, 0.98, 0.80). Even better, if we add the
# padding right before the equality sign, ...

class Dataset:
    def __init__(self, base, number_length, pre_end_padding=0,
                 flip=False, preferred_dtype='int64'):
        self.base = base
        self.number_length = number_length
        self.pre_end_padding = pre_end_padding
        self.flip = flip
        self.preferred_dtype = preferred_dtype

        self.start_token = base  # Before input
        self.end_token = base + 1  # After input
        self.separator_token = base + 2
        self.padding_token = base + 3  # Before input and after target
        self.eos_token = base + 4  # After target
        self.n_tokens = base + 5

        self.dic = {i: str(i) for i in range(self.base + 1)}
        self.dic[self.padding_token] = ""
        self.dic[self.start_token] = ""
        self.dic[self.end_token] = "="
        self.dic[self.eos_token] = ""

    def make_numbers(self, shape, number_length=None, low=None, high=None):
        if number_length is None:
            number_length = self.number_length
        if low == None:
            low = -10**number_length
        if high==None: 
            high = 10**number_length
        return np.random.randint(low,high,shape)
    
    def make_op(self, shape, mode='add'):
        row,col = shape
        #operation is tuple (op,position)
        #alternative: formulate the entire sequence 
        #then translate it into tokens 
        #to_token function
        #one_step_evaluate function
        #but we still need make_op
        #there will be a sequence of numbers and operations 
        #the interpreter function will take care of parsing everything
        
        #rule: do it the way people do it
        #final clean idea, ops are +,-,* but op_list only include +,*
        #num_list includes negative numbers
        #when generating the lhs we condense +- to - 
        #pos_list is always 1,3,5...
        #when evaluating lhs simple pemdas 
        print('mode: ', mode)
        if mode == 'add':
            #no addition in front of first token
            op_list = self.add_token*np.ones((row,col-1)) 
        elif mode == 'mult': 
            op_list = self.mult_token*np.ones((row,col-1))
        elif mode == 'mult-add':
            #a * b + c + d
            op_list = np.random.choice([1,0], size=(row,col-1))
            op_list = np.where(op_list == 1,self.mult_token,self.add_token) 
        elif mode == 'innerprod': 
            #a*b + c*d 
            if col%2 != 0:
                raise ValueError('inner product must have even number of arguments')
            op_list = np.zeros(col-1)
            skip = np.arange(0,col-1,2,dtype=int) 
            op_list[skip] = 1
            op_list = np.where(op_list == 1,self.mult_token,self.add_token)
            op_list = np.tile(op_list,(row,1))
        else: 
            raise ValueError('mode not handled')
        
        return op_list

    def to_digits(self, numbers, length=None):
        if length is None:
            #this line is buggy TODO: fix
            length = max([len(str(num)) for num in numbers]) 

        # Convert numbers to digits
        tensor = np.tile(np.expand_dims(numbers, 1), (1, length))
        exponents = np.arange(length - 1, -1, -1, dtype=self.preferred_dtype)
        bases = np.expand_dims(np.power(self.base, exponents), 0)
        digits = (tensor // bases) % self.base

        # Mask leading zeros
        mask = digits.cumsum(1) == 0
        mask[:, -1] = False
        digits[mask] = self.padding_token
        if self.flip:
            return np.flip(digits, [1])
        return digits

    #this function is useful in rapidly generating data with np.concat
    def move_padding_to_end(self, tensor, end=True):
        """Move all padding tokens in each row to the end without reordering the rest."""

        # Create a tensor with large values where there's padding and row-wise indices elsewhere
        # This allows us to "sort" the padding to the end, while keeping everything else in its
        # original order.
        sorting_tensor = np.where(
            tensor == self.padding_token,
            tensor.shape[1] if end else -tensor.shape[1],
            np.arange(tensor.shape[1])
        )

        # Get the indices that would sort the tensor
        sorted_indices = np.argsort(sorting_tensor, axis=1)

        # Use the sorted indices to rearrange the original tensor
        sorted_tensor = np.take_along_axis(tensor, sorted_indices, 1)

        return sorted_tensor

    def generate_batch(self, bs, **kwargs):
        mode = kwargs['mode']
        sign = kwargs['sign']
        res = self._generate_batch(bs, mode=mode, sign=sign)
        res = self.move_padding_to_end(res)

        # Insert COT padding
        if self.pre_end_padding != 0:
            indices_padding = (res == self.end_token).nonzero(as_tuple=True)
            expanded_tensor = torch.zeros(bs, self.seq + self.pre_end_padding, dtype=res.dtype)
            # Calculate the positions in the expanded tensor for all elements
            positions = torch.arange(self.seq).unsqueeze(0).repeat(bs, 1)
            positions += self.pre_end_padding * (positions >= indices_padding[1].unsqueeze(1))
            # Use scatter to insert values at the correct positions
            expanded_tensor.scatter_(1, positions, res)
            res = expanded_tensor  
        return res

    def _generate_batch(self, tokens):
        assert False, "Not implemented"

    def repr_example(self, example):
        tokens = [
            (tuple(group)[::-1] if self.flip else tuple(group))
            if is_number
            else next(group)
            for is_number, group in itertools.groupby(
                example.tolist(), key=lambda x: x < self.base
            )
        ]
        return self._repr_tokens(tokens).strip()

    def _repr_tokens(self, tokens):
        res = []
        for token in tokens:
            if type(token) is tuple:
                res.append("".join(map(str, token)))
            else:
                res.append(self.dic[token])
        return " ".join(res)

    @property
    def seq(self):
        assert False, "Not implemented"


class BasicOpDataset(Dataset):
    #number of args, number of digits, add/minus/mixed
    def __init__(
        self,
        number_length,
        base=10,
        num_args=4,
        pre_end_padding=0,
        min_b=0,
        flip=False,
        **kwargs,
    ):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)


        self.dic[base+4] = '+'
        self.dic[base+5] = '*'
        self.dic[base+6] = '-'
        self.min_b = min_b

        self.start_token = base  # Before input
        self.end_token = base + 1  # After input
        #self.separator_token = base + 2
        self.padding_token = base + 2  # Before input and after target
        self.eos_token = base + 3  # After target

        self.add_token = base + 4  # between inputs add token
        self.mult_token = base + 5 
        self.minus_token = base + 6
        self.n_tokens = base + 7
        #how to add negative tokens? Token count starts at zero?  
        #this is annoying 
        #this choice can come to bite you
        #minus is a separate operator 
        self.num_args = num_args

    def _generate_batch(self, bs, mode='add', sign='pos'):
        #ode = 'add'
        #ign = 'pos-neg'
        
        print('mode first gen batch: ', mode)
        #mode = {'add','minus', 'add-minus','mult'}
        #'add' every number is positive
        #'minus' every number is negative 
        #'add-minus' numbers can be pos or neg
        #'mult' ought to only ever take two args 
        #'mult' should be trained on general add/minus 
        
        #consistency: each class creates single type of dataset 
        #batch size entirely set by curriculum
        #number of tokens and token dictionary 
        
        #if mode == 'mult' and self.num_args > 2:
        #    raise NotImplemented('multiplication of more than two numbers not yet implemented')
        
        #Question: Need to fix n_tokens beforehand for pretraining to be meaningful
        #for now just deal with subtraction 
        
        #set batch size so that after concatenate we have correct size of data
        if self.num_args%2 == 1:
            raise ValueError("num_args must be even")
        
        #length is number of digits 
        #a+b+c = eval(a+b) + c = eval(a+b+c)
        #TODO: num list is oriented row by column as intuitively
        #for the data_sum function. Change the orientation in previous functions 
        
        #low inclusive high exclusive
        #two sets of flags 
        #add, mult, innerprod for the operation between two numbers
        #pos, neg, pos-neg 
        if sign == 'pos': 
            low = 0
            high = 10**self.number_length
        elif sign == 'neg': 
            low = -10**self.number_length + 1
            high = 0
        elif sign == 'pos-neg': 
            low = -10**self.number_length + 1
            high = 10**self.number_length
        else: 
            raise NotImplemented

        low = int(low)
        high = int(high)
        num_list = self.make_numbers((bs, self.num_args),low=low,high=high)
        print('mode in gen batch: ', mode)
        op_list = self.make_op((bs, self.num_args),mode=mode)
        
        print('num_list: ', num_list[:4,:])
        print('op_list: ', op_list[:4,:])
        
        #raise ValueError('End of implementation') 
        #mult is different than add
        #minus is not an operation 
        #-3 and 3 are two separate tokens so 3 + (-3) = 0.  
        #mult and plus are the operations between any two numbers 
        #this would capture inner product in an elegant way
        #operations array, then scan through operations for pemdas to form rhs 
        #raise ValueError('done')
        
        #params can have dictionary argument "if-then-else" "var substitution etc." 
        #TODO: completely not implemented
        params = {'arithmetic': (num_list, op_list)}
        expr_type = 'arithmetic'
        data = self.gen_data(expr_type,params)
        
        np.random.shuffle(data)
        self.pretty_print(data[:4,:])
        data = data[:bs,:]
        return data

    @property
    def seq(self):
        return 4*self.number_length * self.num_args 
    
BasicOpDataset.gen_data = gen_data
BasicOpDataset.gen_lhs = gen_lhs
BasicOpDataset.gen_rhs = gen_rhs
BasicOpDataset.update = update
BasicOpDataset.combine = combine
BasicOpDataset.num_to_digits = num_to_digits
BasicOpDataset.one_step = one_step
BasicOpDataset.gen_equation = gen_equation
BasicOpDataset.gen_line = gen_line 
BasicOpDataset.pretty_print = pretty_print
        
class InnerProductDataset(Dataset):
    def __init__(
        self,
        number_length,
        base=10,
        num_args=4,
        pre_end_padding=0,
        min_b=0,
        flip=False,
        **kwargs,
    ):
        super().__init__(base, number_length, pre_end_padding, flip, **kwargs)


        self.dic[base+4] = '+'
        self.dic[base+5] = '*'
        self.dict[base+6] = '-'
        self.min_b = min_b

        self.start_token = base  # Before input
        self.end_token = base + 1  # After input
        #self.separator_token = base + 2
        self.padding_token = base + 2  # Before input and after target
        self.eos_token = base + 3  # After target

        self.add_token = base + 4  # between inputs add token
        self.mult_token = base + 5 
        self.minus_token = base + 6
        self.n_tokens = base + 7
        
        self.num_args = num_args

    def _generate_batch(self, bs):
        #set batch size so that after concatenate we have correct size of data
        bs_small = int(bs/2)
        #print('bs_small: ', bs_small)
        if self.num_args%2 == 1:
            raise ValueError("num_args must be even")
        if self.num_args < 4: 
            raise ValueError('At least four digits required for recursive pemdas')
        
        #print('NUM ARGS: ', self.num_args)
        
        #First generate list of numbers 
        num_list = self.make_numbers((self.num_args, bs_small))
        #print('num_list shape: ', np.shape(num_list))
        #generate batch function 
        #take RHS of data and copy it to LHS
        
        #TODO: Insert multiplication and add datasets
    
        #construct lhs a*b + c*d
        data = self.innerprod_lhs(num_list, bs_small)
        #construct rhs eval(a*b) + c*d
        data = self.innerprod_rhs(num_list,bs_small,data,pointer=2)
        #construct lhs eval(a*b) + c*d copies from rhs
        data_inter = self.innerprod_lhs(num_list,bs_small,data=data)
        #construct rhs eval(a*b) + eval(c*d) 
        data_final = self.innerprod_rhs(num_list,bs_small,data_inter,pointer=4)
        #TODO: recursive sum 
        #recursive_sum(num_list)
        
        #length is number of digits in addition
        #a+b+c = eval(a+b) + c = eval(a+b+c)
        #TODO: num list is oriented row by column as intuitively
        #for the data_sum function. Change the orientation in previous functions 
        num_list = self.make_numbers((bs_small, self.num_args), number_length=1)
        #print('num_list: ', num_list)
        data_sum = self.sum_data(num_list)
        #data_mult = self.mult_data(num_list,bs)
        
        #Consider shuffling data
        #np.random.shuffle(data)
        #print('data: ', data[:4,:])
        #print('data_final: ', data_final[:4,:])
        #print('data sum: ', data_sum[:4,:])
        self.pretty_print(data[:4,:])
        self.pretty_print(data_final[:4,:])
        self.pretty_print(data_sum[:4,:])
        data_out = np.concatenate([data,data_final,data_sum],axis=0)
        np.random.shuffle(data_out)
        data_out = data_out[:bs,:]
        return data_out

    @property
    def seq(self):
        return 4*self.number_length * self.num_args 

InnerProductDataset.innerprod_lhs = innerprod_lhs 
InnerProductDataset.innerprod_rhs = innerprod_rhs 
InnerProductDataset.sum_data = sum_data
InnerProductDataset.pretty_print = pretty_print
