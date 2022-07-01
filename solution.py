# board = [
#    [7,8,0,4,0,0,1,2,0],
#    [6,0,0,0,7,5,0,0,9],
#    [0,0,0,6,0,1,0,7,8],
#    [0,0,7,0,4,0,2,6,0],
#    [0,0,1,0,5,0,9,3,0],
#    [9,0,4,0,6,0,0,0,5],
#    [0,7,0,3,0,0,0,1,2],
#    [1,2,0,0,0,7,4,0,0],
#    [0,4,9,2,0,6,0,0,7]]


def solveSudoku(board):
    empty_space = find_empty_space(board)
    if empty_space:
        row, col = empty_space
    else:
        return True
    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i
            if not solveSudoku(board):
                board[row][col] = 0
            else:
                return True
    return False


def find_empty_space(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return i, j
    return None


def valid(board, num, pos):
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    board_x = pos[1] // 3
    board_y = pos[0] // 3
    for i in range(board_y*3, board_y*3+3):
        for j in range(board_x*3, board_x*3+3):
            if board[i][j] == num and (i, j) != pos:
                return False
    return True


def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - ")
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")

# print_board(board)
# solveSudoku(board)
# print("___________________")
# print_board(board)