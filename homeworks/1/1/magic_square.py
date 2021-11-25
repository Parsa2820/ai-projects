import copy

class MagicSquareSearcher:
    def __init__(self, square, size):
        self.square = square
        self.size = size
        self.nine_place = self.find_nine()

    def find_nine(self):
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.square[i][j] == 9:
                    return i, j
        return -1, -1

    def search(self, history=[]):
        if self.is_magic():
            return self.square
        if self.square in history:
            return
        history.append(copy.deepcopy(self.square))
        u_result = self.swap_up_search()
        d_result = self.swap_down_search()
        r_result = self.swap_right_search()
        l_result = self.swap_left_search()
        history.pop()
        if u_result:
            return u_result
        if d_result:
            return d_result
        if r_result:
            return r_result
        if l_result:
            return l_result
        return None

    def swap_up_search(self):
        x, y = self.nine_place
        if x == 0:
            return
        self.square[x][y] = self.square[x-1][y]
        self.square[x-1][y] = 9
        self.nine_place = x-1, y
        ans = self.search()
        self.square[x-1][y] = self.square[x][y]
        self.square[x][y] = 9
        self.nine_place = x, y
        return ans

    def swap_down_search(self):
        x, y = self.nine_place
        if x == self.size-1:
            return
        self.square[x][y] = self.square[x+1][y]
        self.square[x+1][y] = 9
        self.nine_place = x+1, y
        ans = self.search()
        self.square[x+1][y] = self.square[x][y]
        self.square[x][y] = 9
        self.nine_place = x, y
        return ans

    def swap_right_search(self):
        x, y = self.nine_place
        if y == self.size-1:
            return
        self.square[x][y] = self.square[x][y+1]
        self.square[x][y+1] = 9
        self.nine_place = x, y+1
        ans = self.search()
        self.square[x][y+1] = self.square[x][y]
        self.square[x][y] = 9
        self.nine_place = x, y
        return ans

    def swap_left_search(self):
        x, y = self.nine_place
        if y == 0:
            return
        self.square[x][y] = self.square[x][y-1]
        self.square[x][y-1] = 9
        self.nine_place = x, y-1
        ans = self.search()
        self.square[x][y-1] = self.square[x][y]
        self.square[x][y] = 9
        self.nine_place = x, y
        return ans

    def is_magic(self):
        magic_sum = self.get_magic_sum()
        return self.check_row_col(magic_sum)
        return self.check_row_col(magic_sum) and self.check_diagonal(magic_sum) 

    def get_magic_sum(self):
        value = 0
        for i in range(0, self.size):
            value += self.square[0][i]
        return value

    def check_row_col(self, sum):
        for i in range(0, self.size):
            tmp_value_row = 0
            tmp_value_col = 0
            for j in range(0, self.size):
                tmp_value_row += self.square[i][j]
                tmp_value_col += self.square[j][i]
            if tmp_value_row != sum or tmp_value_col != sum:
                return False
        return True

    def check_diagonal(self, sum):
        tmp_upper_left_to_lower_right = 0
        tmp_other = 0
        for i in range(0, self.size):
            tmp_upper_left_to_lower_right += self.square[i][i] 
            tmp_other = self.square[self.size-i][i]
        return tmp_upper_left_to_lower_right == sum and tmp_other == sum

def main():
    square = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    searcher = MagicSquareSearcher(square, 3)
    print(searcher.search())

if __name__ == '__main__':
    main()