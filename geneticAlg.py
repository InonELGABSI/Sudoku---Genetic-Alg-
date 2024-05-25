import random

class Sudoku:
  def __init__(self, board_size,board=None):
    self.size = board_size
    if board!=None:
      self.board=board
    else:
      self.board = [[-1 for _ in range(board_size)] for _ in range(board_size)]
  
  def is_valid(self, row, col, val):
    # Check row and column (excluding the selected cell)
    for i in range(self.size):
      if i != col and self.board[row][i][0] == val:  # Check value (index 0)
        return False
    for i in range(self.size):
      if i != row and self.board[i][col][0] == val:  # Check value (index 0)
        return False

    # Check subgrid (excluding the selected cell)
    subgrid_size = int(self.size**0.5)
    start_row = row - row % subgrid_size
    start_col = col - col % subgrid_size
    for i in range(subgrid_size):
      for j in range(subgrid_size):
        if start_row + i != row and start_col + j != col and self.board[start_row + i][start_col + j][0] == val:
          return False
    return True
  
  def not_in_place (self):
    count = 0
    for i in range(self.size):
      for j in range(self.size):
        if not self.board[i][j][1] and not self.is_valid(i,j,self.board[i][j][0]):
          count+=1
    return count

  def set_board(self, board):
    self.board = [[ (board[i][j],board[i][j]!=0) for i in range(0,self.size)] for j in range(0,self.size)]
  
  def set_tuple_board(self, board):
    self.board = [[ board[i+9*j] for i in range(0,self.size)] for j in range(0,self.size)]

  def get_fitness(self):
    penalty = 0
    for row in range(self.size):
      for col in range(self.size):
        val,flag = self.board[row][col]
        if val != -1 and not flag and not self.is_valid(row, col, val):
          penalty += 1
    return 1 / (penalty + 1)  # Higher score for fewer penalties

class Chromosome:
  def __init__(self, sudoku):
    self.sudoku = sudoku  # Store the entire Sudoku board state
    self.fitness = None

  def calculate_fitness(self):
    self.fitness = self.sudoku.get_fitness()
##############################################

def generate_population(sudoku, population_size):
  population = []
  valid_values = [i for i in range(1, sudoku.size + 1)]

  for _ in range(population_size):
    chromosome = []
    for row in range(sudoku.size):
      for col in range(sudoku.size):
        val,flag = sudoku.board[row][col]
        # Set flag based on initial board value
        
        if val != 0:
          chromosome.append((val, True))
        else:
          # Fill remaining slots with random valid values
          chromosome.append((random.choice(valid_values), False))
    s =Sudoku(sudoku.size)
    s.set_tuple_board(chromosome)
    population.append(Chromosome(s))
  return population


def tournament_selection(population, tournament_size):
  selected = []
  for _ in range(tournament_size):
    competitors = random.sample(population, tournament_size)
    best = max(competitors, key=lambda x: x.fitness)
    selected.append(best)
  return selected


import random

def single_point_crossover(parent1, parent2):
  # Determine crossover point within the valid range (all positions)
  crossover_point = random.randint(0, len(parent1.sudoku.board[0]) - 1)

  # Create child Sudoku objects with inherited fixed elements (first column)
  child1 = Chromosome(Sudoku(len(parent1.sudoku.board)))
  child2 = Chromosome(Sudoku(len(parent2.sudoku.board)))

  # Alternate crossover for remaining elements in each row
  for i in range(0, len(parent1.sudoku.board)):
    # Child 1: First half from parent1, second half from parent2
    for j in range(crossover_point + 1):  # Include crossover point in child1
      if i%2==0:
        child1.sudoku.board[i][j] = parent1.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent2.sudoku.board[i][j]
      else:
        child1.sudoku.board[i][j] = parent2.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent1.sudoku.board[i][j]

    for j in range(crossover_point + 1, len(parent1.sudoku.board[0])):
      if i%2==0:
        child1.sudoku.board[i][j] = parent2.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent1.sudoku.board[i][j]
      else:
        child1.sudoku.board[i][j] = parent1.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent2.sudoku.board[i][j]

  return child1, child2

def crossover(parent1, parent2):
  # Determine crossover point within the valid range (all positions)
  crossover_point = random.randint(0, len(parent1.sudoku.board[0]) - 1)

  # Create child Sudoku objects with inherited fixed elements (first column)
  child1 = Chromosome(Sudoku(len(parent1.sudoku.board)))
  child2 = Chromosome(Sudoku(len(parent2.sudoku.board)))

  # Alternate crossover for remaining elements in each row
  for i in range(0, len(parent1.sudoku.board)):
    # Child 1: First half from parent1, second half from parent2
    for j in range(crossover_point + 1):  # Include crossover point in child1
        child1.sudoku.board[i][j] = parent1.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent2.sudoku.board[i][j]
     

    for j in range(crossover_point + 1, len(parent1.sudoku.board[0])):
        child1.sudoku.board[i][j] = parent2.sudoku.board[i][j]
        child2.sudoku.board[i][j] = parent1.sudoku.board[i][j]

  return child1, child2

def swap_mutation(chromosome, mutation_rate):
  for row in range(len(chromosome.sudoku.board)):
    if random.random() < mutation_rate:
      # Randomly select two cells within the Sudoku board
      col1 = random.randint(0, len(chromosome.sudoku.board[0]) - 1)
      col2 = random.randint(0, len(chromosome.sudoku.board[0]) - 1)
      
      # Check if both cells are mutable (not fixed values)
      while chromosome.sudoku.board[row][col1][1] or chromosome.sudoku.board[row][col2][1]:
        # If one or both are fixed, regenerate random columns
        col1 = random.randint(0, len(chromosome.sudoku.board[0]) - 1)
        col2 = random.randint(0, len(chromosome.sudoku.board[0]) - 1)
      
      # Swap the values only if both cells are mutable
      chromosome.sudoku.board[row][col1], chromosome.sudoku.board[row][col2] = chromosome.sudoku.board[row][col2], chromosome.sudoku.board[row][col1]


def solve_sudoku(sudoku, param_ranges):
  best_params, best_fitness = optimize_parameters(sudoku, param_ranges)
  population_size, generations, tournament_size, mutation_rate = best_params
  
  solved_sudoku = _solve_sudoku_inner(sudoku, population_size, generations, tournament_size, mutation_rate)
  return best_params,solved_sudoku

def create_offspring(parents, mutation_rate):

  offspring = []
  num_parents = len(parents)

  for i in range(0, num_parents, 2):
    # Access parents (considering even or odd case)
    parent1 = parents[i]
    parent2 = None
    if i + 1 < num_parents:
      parent2 = parents[i + 1]

    # Perform operations only if a second parent exists
    if parent2 is not None:
      child1, child2 = crossover(parent1, parent2)
      swap_mutation(child1, mutation_rate)
      swap_mutation(child2, mutation_rate)
      offspring.append(child1)
      offspring.append(child2)
    else:
      # Handle odd number: apply mutation directly to last unpaired parent
      swap_mutation(parent1, mutation_rate)
      offspring.append(parent1)  # Copy the unpaired parent

  return offspring


def _solve_sudoku_inner(sudoku, population_size, generations, tournament_size, mutation_rate):
  population = generate_population(sudoku, population_size)
   # Calculate fitness for all chromosomes in initial population
  for chromosome in population:
    chromosome.calculate_fitness()
  for _ in range(generations):
    parents = tournament_selection(population, tournament_size)
    population = create_offspring(parents,mutation_rate)
    for chromosome in population:
      chromosome.calculate_fitness()

  best_chromosome = max(population, key=lambda x: x.fitness)
  population=[]
  return best_chromosome.sudoku



def optimize_parameters(sudoku, param_ranges):

  best_params = None
  best_fitness = float('-inf')  # Negative infinity for maximization

  for population_size in range(param_ranges['population_size'][0], param_ranges['population_size'][1] + 1):
    for generations in range(param_ranges['generations'][0], param_ranges['generations'][1] + 1):
      for tournament_size in range(param_ranges['tournament_size'][0], param_ranges['tournament_size'][1] + 1):
        for mutation_rate in range(param_ranges['mutation_rate'][0], param_ranges['mutation_rate'][1] + 1):  # Discretize mutation rate (0.00 to 1.00)
          mutation_rate/=100
          solved_sudoku = _solve_sudoku_inner(sudoku, population_size, generations, tournament_size, mutation_rate)
          fitness = solved_sudoku.get_fitness()
          if fitness > best_fitness:
            best_fitness = fitness
            best_params = (population_size, generations, tournament_size, mutation_rate)
  
  return best_params, best_fitness



# Example usage
sudoku_size = 9  # Assuming a 9x9 Sudoku
solution_board = [
  [0, 9, 3, 1, 7, 5, 6, 4, 2],
  [7, 2, 4, 8, 3, 6, 9, 1, 5],
  [5, 6, 1, 2, 4, 9, 3, 8, 7],
  [2, 1, 5, 6, 8, 4, 7, 9, 3],
  [4, 3, 6, 9, 1, 7, 5, 2, 8],
  [9, 7, 8, 5, 2, 3, 4, 6, 1],
  [3, 5, 2, 4, 6, 8, 1, 7, 9],
  [6, 8, 9, 7, 5, 1, 2, 3, 4],
  [1, 4, 7, 3, 9, 2, 8, 5, 6]
]
initial_board = [
  [0, 9, 3, 1, 0, 5, 6, 4, 0],
  [7, 0, 0, 0, 0, 0, 0, 0, 5],
  [5, 0, 1, 2, 0, 9, 3, 0, 7],
  [2, 0, 0, 0, 0, 0, 0, 0, 3],
  [0, 3, 6, 9, 0, 7, 5, 2, 0],
  [9, 0, 0, 0, 0, 0, 0, 0, 1],
  [3, 0, 2, 4, 0, 8, 1, 0, 9],
  [6, 0, 0, 0, 0, 0, 0, 0, 4],
  [0, 4, 7, 3, 0, 2, 8, 5, 0]
]

board=[[ (solution_board[i][j],initial_board[i][j]==solution_board[i][j]) for i in range(0,sudoku_size)] for j in range(0,sudoku_size)]
sudoku = Sudoku(sudoku_size,board)

print(sudoku.get_fitness())
print(sudoku.not_in_place())
# Example usage
sudoku_size = 9  # Assuming a 9x9 Sudoku


sudoku = Sudoku(sudoku_size)
sudoku.set_board(initial_board)

# Define parameter ranges for optimization (adjust as needed)
param_ranges = {
  'population_size': (1000, 1000),
  'generations': (100, 100),
  'tournament_size': (100, 100),
  'mutation_rate': (1, 1) # 1=0.01
}

best_params,solved_sudoku = solve_sudoku(sudoku, param_ranges)

# Print the solved Sudoku board (assuming solved_sudoku.board is a 2D list)
for row in solved_sudoku.board:
  print([x for x,_ in row])
print(solved_sudoku.get_fitness())
print(solved_sudoku.not_in_place())
print(best_params)
