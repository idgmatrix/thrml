import sys
import numpy as np
from random import shuffle, random, sample, randint
from copy import deepcopy
from math import exp

class SudokuPuzzle(object):
    def __init__(self, data=None, original_entries=None):
        """
                        data - input puzzle as one array, all rows concatenated.
                               (default - incomplete puzzle)   
        
            original_entries - for inheritance of the original entries of one
                                sudoku puzzle's original, immutable entries we don't 
                                allow to change between random steps.                       
        """
        if data is None:
            self.data = np.array([5,3,0,0,7,0,0,0,0,
                                  6,0,0,1,9,5,0,0,0,
                                  0,9,8,0,0,0,0,6,0,
                                  8,0,0,0,6,0,0,0,3,
                                  4,0,0,8,0,3,0,0,1,
                                  7,0,0,0,2,0,0,0,6,
                                  0,6,0,0,0,0,2,8,0,
                                  0,0,0,4,1,9,0,0,5,
                                  0,0,0,0,8,0,0,7,9])
        else:
            self.data = data
    
        if original_entries is None:
            self.original_entries = np.arange(81)[self.data > 0]
        else:
            self.original_entries = original_entries
            
    def randomize_on_zeroes(self):
        """
        Go through entries, replace incomplete entries (zeroes) 
        with random numbers.
        """
        for num in range(9):
            block_indices = self.get_block_indices(num)
            block = self.data[block_indices]
            zero_indices = [ind for i,ind in enumerate(block_indices) if block[i] == 0]
            to_fill = [i for i in range(1,10) if i not in block]
            shuffle(to_fill)
            for ind, value in zip(zero_indices, to_fill):
                self.data[ind] = value
            
    def get_block_indices(self, k, ignore_originals=False):
        """
        Get data indices for kth block of puzzle.
        """
        row_offset = (k // 3) * 3
        col_offset = (k % 3)  * 3
        indices = [col_offset + (j%3) + 9*(row_offset + (j//3)) for j in range(9)]
        if ignore_originals:
            indices = list(filter(lambda x:x not in self.original_entries, indices))
        return indices
        
    def get_column_indices(self, i, type="data index"):
        """
        Get all indices for the column of ith index
        or for the ith column (depending on type)
        """
        if type=="data index":
            column = i % 9
        elif type=="column index":
            column = i
        else:
            raise ValueError("Invalid type parameter")
        indices = [column + 9 * j for j in range(9)]
        return indices

    def get_row_indices(self, i, type="data index"):
        """
        Get all indices for the row of ith index
        or for the ith row (depending on type)
        """
        if type=="data index":
            row = i // 9
        elif type=="row index":
            row = i
        else:
            raise ValueError("Invalid type parameter")
        indices = [j + 9*row for j in range(9)]
        return indices
        indices = [j + 9*row for j in range(9)]
        return indices
        
    def view_results(self):
        """
        Visualize results as a 9 by 9 grid 
        (given as a two-dimensional numpy array)
        """
        def notzero(s):
            if s != 0: return str(s)
            if s == 0: return "'"
            
        results = np.array([self.data[self.get_row_indices(j, type="row index")] for j in range(9)])
        out_s = ""
        for i, row in enumerate(results):
            if i%3==0: 
                out_s += "="*25+'\n'
            out_s += "| " + " | ".join([" ".join(notzero(s) for s in list(row)[3*(k-1):3*k]) for k in range(1,4)]) + " |\n"
        out_s += "="*25+'\n'
        print(out_s)

    def score_board(self):
        """
        Score board by viewing every row and column and giving
        -1 points for each unique entry.
        """
        score = 0
        for row in range(9):
            score -= len(set(self.data[self.get_row_indices(row, type="row index")]))
        for col in range(9):
            score -= len(set(self.data[self.get_column_indices(col,type="column index")]))
        return score
        
    def make_candidate_data(self):
        """
        Generates "neighbor" board by randomly picking
        a square, then swapping two small squares within.
        """
        new_data = deepcopy(self.data)
        block = randint(0,8)
        num_in_block = len(self.get_block_indices(block, ignore_originals=True))
        random_squares = sample(range(num_in_block),2)
        square1, square2 = [self.get_block_indices(block, ignore_originals=True)[ind] for ind in random_squares]
        new_data[square1], new_data[square2] = new_data[square2], new_data[square1]
        return new_data

def sudoku_solver(input_data=None, mode='default'):
    """
    Uses a simulated annealing technique to solve a Sudoku puzzle.

    Randomly fills out the sub-squares to be consistent sub-solutions.

    Scores a puzzle by giving a -1 for every unique element
    in each row or each column. Best solution has a score of -162.
    (This is our stopping rule.)

    Candidate for new puzzle is created by randomly selecting
    sub-square, then randomly flipping two of its entries, evaluating
    the new score. The delta_S is the difference between the scores.

    Let T be the global temperature of our system, with a geometric
    schedule for decreasing (perhaps by T <- .999 T).

    If U is drawn uniformly from [0,1], and exp((delta_S/T)) > U,
    then we accept the candidate solution as our state.
    """

    if input_data is not None:
        sp = SudokuPuzzle(input_data)
    elif mode == 'empty':
        sp = SudokuPuzzle(np.zeros(81, dtype=int))
    else:
        sp = SudokuPuzzle()

    print("Original Puzzle:")
    sp.view_results()
    sp.randomize_on_zeroes()
    best_sp = deepcopy(sp)
    current_score = sp.score_board()
    best_score = current_score
    t = .5
    count = 0

    while (count < 400000):
        try:
            if (count % 1000 == 0):
                print("Iteration %s,    \tt = %.5f, \tbest_score = %s, \tcurrent_score = %s"%(count, t,
                                                               best_score, current_score))
            candidate_data = sp.make_candidate_data()
            sp_candidate = SudokuPuzzle(candidate_data, sp.original_entries)
            candidate_score = sp_candidate.score_board()
            delta_S = float(current_score - candidate_score)

            if (exp((delta_S/t)) - random() > 0):
                sp = sp_candidate
                current_score = candidate_score

            if (current_score < best_score):
                best_sp = deepcopy(sp)
                best_score = best_sp.score_board()

            if candidate_score == -162:
                sp = sp_candidate
                break

            t = .99999*t
            count += 1
        except:
            print("Hit an inexplicable numerical error. It's a random algorithm-- try again.")
    if best_score == -162:
        print("\nSOLVED THE PUZZLE.")
    else:
        print("\nDIDN'T SOLVE. (%s/%s points). It's a random algorithm-- try again."%(best_score,-162))
    print("\nFinal Puzzle:")
    sp.view_results()

if __name__ == "__main__":
    mode = 'default'
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    elif len(sys.argv) == 1:
        mode_input = input("Choose mode: 1 for partial filled, 2 for all empty: ").strip()
        if mode_input == '2':
            mode = 'empty'
        else:
            mode = 'default'

    if len(sys.argv) > 1:
        try:
            input_puzzle = np.array([int(s) for s in sys.argv[1]])
            assert len(input_puzzle) == 81, "Puzzle must have 81 entries."
            sudoku_solver(input_data=input_puzzle, mode=mode)
        except ValueError:
            print("Puzzle must be 81 consecutive integers, 0s for skipped entries.")
        except AssertionError as e:
            print(e)
    else:
        sudoku_solver(mode=mode)
        
