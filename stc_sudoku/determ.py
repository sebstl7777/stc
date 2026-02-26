import numpy as np
from sudoku import Sudoku
from stc_sudoku.funcs import run_stc_solver

def display_board(board_array, size):
    """Helper to display numpy arrays using the py-sudoku package."""
    box_size = int(np.sqrt(size))
    # py-sudoku uses 0 to represent empty cells
    board_list = board_array.tolist()
    puzzle = Sudoku(box_size, box_size, board=board_list)
    puzzle.show()

def main():
    # ==========================================
    # 4x4 SUDOKU TEST (2x2 boxes)
    # ==========================================
    print("========================================")
    print("TESTING 4x4 BOARD")
    print("========================================")
    
    # A simple 4x4 puzzle (0 is empty)
    puzzle_4x4 = np.array([
        [0, 3, 4, 0],
        [4, 0, 0, 2],
        [1, 0, 0, 3],
        [0, 2, 1, 0]
    ])
    
    print("Initial 4x4 Puzzle:")
    display_board(puzzle_4x4, size=4)
    
    solution_4x4 = run_stc_solver(size=4, given_puzzle=puzzle_4x4, iterations=100, batch_size=500)
    
    print("\nFinal 4x4 Output:")
    display_board(solution_4x4, size=4)


    # ==========================================
    # 9x9 SUDOKU TEST (3x3 boxes)
    # ==========================================
    print("\n========================================")
    print("TESTING 9x9 BOARD")
    print("========================================")
    
    # Completely empty 9x9 board (Pure Generative Mode)
    empty_9x9 = np.zeros((9, 9), dtype=int)
    
    print("Initial 9x9 Puzzle (Empty):")
    display_board(empty_9x9, size=9)
    
    # Needs a larger batch size and more iterations to navigate the larger space
    solution_9x9 = run_stc_solver(size=9, given_puzzle=empty_9x9, iterations=800, batch_size=4000)
    
    print("\nFinal 9x9 Output:")
    display_board(solution_9x9, size=9)

if __name__ == "__main__":
    main()
