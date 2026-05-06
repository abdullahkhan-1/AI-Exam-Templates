"""
╔══════════════════════════════════════════════════════════════════╗
║           AI LAB EXAM - READY-TO-USE CODE TEMPLATES             ║
║   Topics: Adversarial Search | Bayesian/Markov | CSP | ML/EDA   ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE:
- Read the scenario question
- Find the matching section below (CTRL+F the topic)
- Copy the template, change variable names / values to match scenario
- Run and submit

SECTIONS:
  [1] MINIMAX (Tic-Tac-Toe style)
  [2] MINIMAX + ALPHA-BETA PRUNING (Coin-Row / Game Tree)
  [3] DEPTH-LIMITED MINIMAX WITH HEURISTIC
  [4] BAYESIAN NETWORK (pgmpy)
  [5] MARKOV MODEL / CHAIN
  [6] CSP with OR-Tools CP-SAT
  [7] EDA (Exploratory Data Analysis)
  [8] ML ALGORITHMS (Supervised + Unsupervised)
"""

# ═══════════════════════════════════════════════════════════════════
# [1]  MINIMAX ── Tic-Tac-Toe  (works for ANY 3x3 board game)
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "Tic-Tac-Toe", "never lose", "optimal move",
                   "AI player", "board game", "minimax"
CHANGE: nothing for Tic-Tac-Toe. For other games change check_winner()
        and print_board() only.
"""

import math
import copy

# ---------- BOARD HELPERS ----------
def print_board(board):
    for row in board:
        print(" | ".join(cell if cell != " " else " " for cell in row))
        print("-" * 9)

def check_winner(board, player):
    # rows, cols, diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)): return True
        if all(board[j][i] == player for j in range(3)): return True
    if all(board[i][i] == player for i in range(3)): return True
    if all(board[i][2-i] == player for i in range(3)): return True
    return False

def is_full(board):
    return all(board[i][j] != " " for i in range(3) for j in range(3))

def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == " "]

# ---------- MINIMAX ----------
def minimax(board, depth, is_maximizing):
    if check_winner(board, "X"):  return 1          # AI wins
    if check_winner(board, "O"):  return -1         # Human wins
    if is_full(board):            return 0          # Draw

    if is_maximizing:   # AI's turn (X)
        best = -math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = "X"
            score = minimax(board, depth + 1, False)
            board[i][j] = " "
            best = max(best, score)
        return best
    else:               # Human's turn (O)
        best = math.inf
        for (i, j) in get_empty_cells(board):
            board[i][j] = "O"
            score = minimax(board, depth + 1, True)
            board[i][j] = " "
            best = min(best, score)
        return best

def best_move(board):
    best_score = -math.inf
    move = None
    for (i, j) in get_empty_cells(board):
        board[i][j] = "X"
        score = minimax(board, 0, False)
        board[i][j] = " "
        print(f"  Move ({i},{j}) → score: {score}")   # shows reasoning
        if score > best_score:
            best_score, move = score, (i, j)
    return move

def play_tictactoe():
    board = [[" "]*3 for _ in range(3)]
    print("=== TIC-TAC-TOE (You=O, AI=X) ===")
    print("Positions: (row,col) from 0-2")

    for turn in range(9):
        print_board(board)
        if turn % 2 == 0:            # Human goes first
            while True:
                try:
                    r, c = map(int, input("Your move (row col): ").split())
                    if board[r][c] == " ": board[r][c] = "O"; break
                    else: print("Cell taken, try again.")
                except: print("Invalid input. Enter row col (e.g. 1 1)")
        else:                         # AI
            print("AI is thinking...")
            r, c = best_move(board)
            board[r][c] = "X"
            print(f"AI plays: ({r},{c})")

        if check_winner(board, "O"): print_board(board); print("You win!"); return
        if check_winner(board, "X"): print_board(board); print("AI wins!"); return

    print_board(board); print("Draw!")

# Uncomment to run:
# play_tictactoe()


# ═══════════════════════════════════════════════════════════════════
# [2]  MINIMAX + ALPHA-BETA PRUNING ── Coin Row / Game Tree
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "coin row", "pick left or right", "alpha-beta",
                   "game tree", "nodes explored", "pruning"
CHANGE: coins list at the bottom.
"""

nodes_minimax   = 0   # counter for plain minimax
nodes_alphabeta = 0   # counter for alpha-beta

def coin_minimax(coins, left, right, is_max):
    """Plain minimax for coin-row game."""
    global nodes_minimax
    nodes_minimax += 1
    if left > right: return 0
    if is_max:
        return max(coins[left]  + coin_minimax(coins, left+1, right, False),
                   coins[right] + coin_minimax(coins, left, right-1, False))
    else:
        return min(coin_minimax(coins, left+1, right, True),
                   coin_minimax(coins, left, right-1, True))

def coin_alphabeta(coins, left, right, is_max, alpha, beta, depth=0):
    """Minimax with Alpha-Beta pruning for coin-row."""
    global nodes_alphabeta
    nodes_alphabeta += 1
    if left > right: return 0

    if is_max:
        best = -math.inf
        for pick, nl, nr in [(coins[left],  left+1, right),
                              (coins[right], left,   right-1)]:
            val = pick + coin_alphabeta(coins, nl, nr, False, alpha, beta, depth+1)
            if val > best: best = val
            alpha = max(alpha, best)
            if beta <= alpha:
                print(f"  ✂ PRUNED at depth {depth} | α={alpha:.0f} β={beta:.0f}")
                break
        return best
    else:
        best = math.inf
        for nl, nr in [(left+1, right), (left, right-1)]:
            val = coin_alphabeta(coins, nl, nr, True, alpha, beta, depth+1)
            if val < best: best = val
            beta = min(beta, best)
            if beta <= alpha:
                print(f"  ✂ PRUNED at depth {depth} | α={alpha:.0f} β={beta:.0f}")
                break
        return best

def run_coin_row(coins):
    global nodes_minimax, nodes_alphabeta
    print(f"\n{'='*50}")
    print(f"Coins: {coins}")

    # Plain Minimax
    nodes_minimax = 0
    score_mm = coin_minimax(coins, 0, len(coins)-1, True)
    print(f"\n[MINIMAX]")
    print(f"  Best score for Max : {score_mm}")
    print(f"  Nodes explored     : {nodes_minimax}")

    # Alpha-Beta
    nodes_alphabeta = 0
    print(f"\n[ALPHA-BETA]")
    score_ab = coin_alphabeta(coins, 0, len(coins)-1, True, -math.inf, math.inf)
    print(f"  Best score for Max : {score_ab}")
    print(f"  Nodes explored     : {nodes_alphabeta}")
    print(f"  Nodes saved        : {nodes_minimax - nodes_alphabeta}")

# ── CHANGE coins here ──
coins_example = [6, 2, 9, 4, 7, 3, 8, 5]
# run_coin_row(coins_example)


# ═══════════════════════════════════════════════════════════════════
# [3]  DEPTH-LIMITED MINIMAX + HEURISTIC  (Cyber/Complex games)
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "depth limit", "heuristic", "too expensive",
                   "cutoff", "cyber", "defender", "attacker"
CHANGE: heuristic() formula, actions list, apply_action() transitions,
        and initial_state values.
"""

# ── State: dictionary of features ──
def heuristic(state):
    """
    CHANGE THIS FORMULA to match scenario.
    Example: Cyber Defense
      health=0-100, vuln=0-10, intrusion=0-10, resources=0-20
    """
    return (2  * state["health"]
          - 8  * state["intrusion"]
          - 5  * state["vuln"]
          + 3  * state["resources"])

# ── Define legal actions & transitions ──
DEFENDER_ACTIONS = ["patch", "scan", "isolate", "monitor"]
ATTACKER_ACTIONS = ["attack", "exploit", "probe", "idle"]

def apply_action(state, action, is_defender):
    """Apply an action and return the new state."""
    s = copy.deepcopy(state)
    if is_defender:
        if action == "patch":    s["vuln"]      = max(0, s["vuln"] - 2)
        if action == "scan":     s["intrusion"] = max(0, s["intrusion"] - 1)
        if action == "isolate":  s["intrusion"] = max(0, s["intrusion"] - 3); s["resources"] -= 3
        if action == "monitor":  s["resources"] = min(20, s["resources"] + 1)
    else:
        if action == "attack":   s["health"]    = max(0, s["health"] - 15)
        if action == "exploit":  s["intrusion"] = min(10, s["intrusion"] + 2)
        if action == "probe":    s["vuln"]      = min(10, s["vuln"] + 1)
        if action == "idle":     pass
    return s

def depth_minimax(state, depth, is_maximizing, alpha=-math.inf, beta=math.inf):
    """Depth-limited minimax with alpha-beta and heuristic cutoff."""
    if depth == 0 or state["health"] <= 0:
        return heuristic(state), None

    actions = DEFENDER_ACTIONS if is_maximizing else ATTACKER_ACTIONS
    best_action = None

    if is_maximizing:
        best_val = -math.inf
        for action in actions:
            new_state = apply_action(state, action, True)
            val, _ = depth_minimax(new_state, depth-1, False, alpha, beta)
            if val > best_val:
                best_val, best_action = val, action
            alpha = max(alpha, best_val)
            if beta <= alpha: break
        return best_val, best_action
    else:
        best_val = math.inf
        for action in actions:
            new_state = apply_action(state, action, False)
            val, _ = depth_minimax(new_state, depth-1, True, alpha, beta)
            if val < best_val:
                best_val, best_action = val, action
            beta = min(beta, best_val)
            if beta <= alpha: break
        return best_val, best_action

def simulate_game(initial_state, turns=5, depth=3):
    """Simulate multiple turns of the game."""
    state = copy.deepcopy(initial_state)
    print(f"Initial State: {state}")
    print(f"Initial Heuristic: {heuristic(state)}\n")

    for turn in range(turns):
        # Defender moves
        val, def_action = depth_minimax(state, depth, True)
        state = apply_action(state, def_action, True)
        print(f"Turn {turn+1} | Defender: {def_action:8s} → state: {state}")

        # Attacker responds
        _, att_action = depth_minimax(state, depth, False)
        state = apply_action(state, att_action, False)
        print(f"Turn {turn+1} | Attacker: {att_action:8s} → state: {state}")
        print(f"         Heuristic score: {heuristic(state)}")

        # Compare depth 3 vs depth 5
        val3, act3 = depth_minimax(state, 3, True)
        val5, act5 = depth_minimax(state, 5, True)
        print(f"         depth=3 suggests: {act3} (score {val3:.1f})")
        print(f"         depth=5 suggests: {act5} (score {val5:.1f})\n")

# ── CHANGE initial state here ──
initial_state = {"health": 80, "vuln": 5, "intrusion": 3, "resources": 10}
# simulate_game(initial_state, turns=5, depth=3)


# ═══════════════════════════════════════════════════════════════════
# [4]  BAYESIAN NETWORK  (pgmpy)
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "Bayesian", "probability", "diagnosis", "disease",
                   "symptoms", "prior", "conditional", "inference",
                   "burglary", "exam performance", "CPT"

CHANGE:
  - edges list  (the DAG structure)
  - TabularCPD values for each node
  - inference.query() variables and evidence
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# ══ TEMPLATE A: Disease Diagnosis (Flu/Cold) ══
def bayesian_disease():
    # Step 1: Define structure (edges = DAG)
    model = DiscreteBayesianNetwork([
        ('Disease', 'Fever'),
        ('Disease', 'Cough'),
        ('Disease', 'Fatigue'),
        ('Disease', 'Chills')
    ])

    # Step 2: Prior — P(Disease)
    # variable_card = number of states
    # values: row 0 = state 0 (Flu), row 1 = state 1 (Cold)
    cpd_disease = TabularCPD(
        variable='Disease', variable_card=2,
        values=[[0.3],   # Flu
                [0.7]])  # Cold

    # Step 3: Conditionals — P(Symptom | Disease)
    # columns order: Disease=Flu, Disease=Cold
    cpd_fever = TabularCPD(
        variable='Fever', variable_card=2,
        values=[[0.1, 0.5],   # Fever=No | Flu, Cold
                [0.9, 0.5]],  # Fever=Yes| Flu, Cold
        evidence=['Disease'], evidence_card=[2])

    cpd_cough = TabularCPD(
        variable='Cough', variable_card=2,
        values=[[0.2, 0.4],
                [0.8, 0.6]],
        evidence=['Disease'], evidence_card=[2])

    cpd_fatigue = TabularCPD(
        variable='Fatigue', variable_card=2,
        values=[[0.3, 0.7],
                [0.7, 0.3]],
        evidence=['Disease'], evidence_card=[2])

    cpd_chills = TabularCPD(
        variable='Chills', variable_card=2,
        values=[[0.4, 0.6],
                [0.6, 0.4]],
        evidence=['Disease'], evidence_card=[2])

    # Step 4: Build model
    model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)
    assert model.check_model(), "Model is invalid!"
    print("Model valid ✓")

    # Step 5: Inference
    infer = VariableElimination(model)

    # Query 1: P(Disease | Fever=Yes, Cough=Yes)
    # 1 = Yes, 0 = No  (matches the order in TabularCPD values rows)
    result1 = infer.query(variables=['Disease'],
                          evidence={'Fever': 1, 'Cough': 1})
    print("\nP(Disease | Fever=Yes, Cough=Yes):")
    print(result1)

    # Query 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)
    result2 = infer.query(variables=['Disease'],
                          evidence={'Fever': 1, 'Cough': 1, 'Chills': 1})
    print("\nP(Disease | Fever=Yes, Cough=Yes, Chills=Yes):")
    print(result2)

    # Query 3: P(Fatigue | Disease=Flu) — Flu=0 in our encoding
    result3 = infer.query(variables=['Fatigue'],
                          evidence={'Disease': 0})
    print("\nP(Fatigue | Disease=Flu):")
    print(result3)

# bayesian_disease()

# ══ TEMPLATE B: Student Exam Performance ══
def bayesian_student():
    model = DiscreteBayesianNetwork([
        ('Intelligence', 'Grade'),
        ('StudyHours',   'Grade'),
        ('Difficulty',   'Grade'),
        ('Grade', 'Pass')
    ])

    cpd_intel = TabularCPD('Intelligence', 2, [[0.3], [0.7]])       # Low, High
    cpd_study = TabularCPD('StudyHours',   2, [[0.4], [0.6]])       # Insuf, Suf
    cpd_diff  = TabularCPD('Difficulty',   2, [[0.6], [0.4]])       # Easy, Hard

    # Grade has 3 states: C=0, B=1, A=2
    # Parents: Intelligence(2) x StudyHours(2) x Difficulty(2) = 8 combos
    # Columns order: I=Low/S=Insuf/D=Easy, I=Low/S=Insuf/D=Hard, ...
    # Each column sums to 1. Rows = [P(C|...), P(B|...), P(A|...)]
    cpd_grade = TabularCPD(
        variable='Grade', variable_card=3,
        values=[
          # C   Low/Insuf/Easy  Low/Insuf/Hard  Low/Suf/Easy  Low/Suf/Hard  Hi/Insuf/Easy  Hi/Insuf/Hard  Hi/Suf/Easy  Hi/Suf/Hard
            [0.6, 0.8,           0.5,             0.7,          0.3,          0.2,            0.1,           0.05],
            [0.3, 0.15,          0.4,             0.25,         0.5,          0.5,            0.4,           0.25],
            [0.1, 0.05,          0.1,             0.05,         0.2,          0.3,            0.5,           0.70],
        ],
        evidence=['Intelligence','StudyHours','Difficulty'],
        evidence_card=[2, 2, 2])

    # Pass: Yes=1, No=0  |  Grade: C=0, B=1, A=2
    cpd_pass = TabularCPD(
        variable='Pass', variable_card=2,
        values=[[0.50, 0.20, 0.05],   # Pass=No  | Grade C,B,A
                [0.50, 0.80, 0.95]],  # Pass=Yes | Grade C,B,A
        evidence=['Grade'], evidence_card=[3])

    model.add_cpds(cpd_intel, cpd_study, cpd_diff, cpd_grade, cpd_pass)
    assert model.check_model()
    print("Student model valid ✓")

    infer = VariableElimination(model)

    # P(Pass | StudyHours=Sufficient=1, Difficulty=Hard=1)
    r1 = infer.query(['Pass'], evidence={'StudyHours': 1, 'Difficulty': 1})
    print("\nP(Pass | StudyHours=Suf, Difficulty=Hard):")
    print(r1)

    # P(Intelligence | Pass=Yes=1)
    r2 = infer.query(['Intelligence'], evidence={'Pass': 1})
    print("\nP(Intelligence | Pass=Yes):")
    print(r2)

# bayesian_student()

# ══ TEMPLATE C: Burglary Alarm (Classic) ══
def bayesian_burglary():
    model = DiscreteBayesianNetwork([
        ('Burglary',   'Alarm'),
        ('Earthquake', 'Alarm'),
        ('Alarm', 'JohnCalls'),
        ('Alarm', 'MaryCalls')
    ])
    cpd_b = TabularCPD('Burglary',   2, [[0.999], [0.001]])
    cpd_e = TabularCPD('Earthquake', 2, [[0.998], [0.002]])
    cpd_a = TabularCPD('Alarm', 2,
        values=[[0.999, 0.71, 0.06, 0.05],
                [0.001, 0.29, 0.94, 0.95]],
        evidence=['Burglary','Earthquake'], evidence_card=[2,2])
    cpd_j = TabularCPD('JohnCalls', 2,
        values=[[0.95, 0.10], [0.05, 0.90]],
        evidence=['Alarm'], evidence_card=[2])
    cpd_m = TabularCPD('MaryCalls', 2,
        values=[[0.99, 0.30], [0.01, 0.70]],
        evidence=['Alarm'], evidence_card=[2])
    model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)
    assert model.check_model()
    infer = VariableElimination(model)
    result = infer.query(['Burglary'], evidence={'JohnCalls': 1, 'MaryCalls': 1})
    print("P(Burglary | John=Yes, Mary=Yes):")
    print(result)

# bayesian_burglary()


# ═══════════════════════════════════════════════════════════════════
# [5]  MARKOV MODEL / CHAIN
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "Markov", "transition", "weather", "states",
                   "simulate", "sequence", "probability of X days"
CHANGE: states, transition_matrix, initial_state, num_steps
"""

import numpy as np

def markov_weather():
    # ── CHANGE these ──
    states = ["Sunny", "Cloudy", "Rainy"]
    # Row = current state, Col = next state; each row must sum to 1
    transition_matrix = np.array([
        [0.6, 0.3, 0.1],  # From Sunny  → Sunny, Cloudy, Rainy
        [0.3, 0.4, 0.3],  # From Cloudy → ...
        [0.2, 0.3, 0.5],  # From Rainy  → ...
    ])
    initial_state = "Sunny"
    num_steps     = 10

    # ── Simulate one sequence ──
    def simulate(init, steps):
        current = init
        seq = [current]
        for _ in range(steps):
            idx = states.index(current)
            current = np.random.choice(states, p=transition_matrix[idx])
            seq.append(current)
        return seq

    sequence = simulate(initial_state, num_steps)
    print(f"State sequence ({num_steps} steps from {initial_state}):")
    print(" → ".join(sequence))

    # ── Monte Carlo: P(at least 3 rainy days) ──
    target_state   = "Rainy"
    target_count   = 3
    num_simulations = 10000
    count = 0
    for _ in range(num_simulations):
        seq = simulate(initial_state, num_steps)
        if seq.count(target_state) >= target_count:
            count += 1
    prob = count / num_simulations
    print(f"\nP(at least {target_count} {target_state} days in {num_steps}) ≈ {prob:.4f}")

    # ── Stationary distribution ──
    state_counts = {s: 0 for s in states}
    long_seq = simulate(initial_state, 100000)
    for s in long_seq: state_counts[s] += 1
    total = len(long_seq)
    print("\nApproximate stationary distribution:")
    for s in states:
        print(f"  P({s}) ≈ {state_counts[s]/total:.4f}")

# markov_weather()

# ── Generic Markov (any domain) ──
def generic_markov(states, transition_matrix, initial_state, num_steps=10):
    """
    states           : list of state names, e.g. ["Red","Blue"]
    transition_matrix: 2D numpy array
    initial_state    : string, must be in states
    """
    current = initial_state
    sequence = [current]
    for _ in range(num_steps):
        idx = states.index(current)
        current = np.random.choice(states, p=transition_matrix[idx])
        sequence.append(current)
    print(" → ".join(sequence))
    return sequence


# ═══════════════════════════════════════════════════════════════════
# [6]  CSP with OR-Tools CP-SAT
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "schedule", "assign", "constraint", "CSP",
                   "N-Queens", "timetable", "resource allocation",
                   "traffic", "no conflict"

CHANGE: variables, domains, constraints for each scenario.
"""

from ortools.sat.python import cp_model

# ══ TEMPLATE A: Exam Timetable ══
def csp_exam_timetable():
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    courses   = ["AI", "DS", "DB", "COAL", "OS"]
    slots     = [1, 2, 3]        # time slots
    rooms     = [1, 2]           # room IDs
    capacity  = {1: 60, 2: 30}  # room capacities
    size      = {"AI":55,"DS":40,"DB":25,"COAL":35,"OS":20}

    # Conflict pairs — cannot be in same slot
    conflicts = [("AI","DS"),("AI","DB"),("DS","COAL"),("DB","OS")]

    # Decision variables: slot and room for each course
    slot_var = {c: model.NewIntVar(1, 3, f"slot_{c}") for c in courses}
    room_var = {c: model.NewIntVar(1, 2, f"room_{c}") for c in courses}

    # No-conflict constraints
    for (a, b) in conflicts:
        model.Add(slot_var[a] != slot_var[b])

    # Capacity constraints
    for c in courses:
        for r in rooms:
            # if course c is in room r, capacity must be enough
            b = model.NewBoolVar(f"in_room_{c}_{r}")
            model.Add(room_var[c] == r).OnlyEnforceIf(b)
            model.Add(room_var[c] != r).OnlyEnforceIf(b.Not())
            if size[c] > capacity[r]:
                model.Add(b == 0)  # cannot use this room

    # Hard rule: COAL must be in slot 1 or 2
    model.AddAllowedAssignments([slot_var["COAL"]], [[1],[2]])

    # Soft preference: AI in slot 1 (minimise violations)
    ai_not_slot1 = model.NewBoolVar("ai_pref")
    model.Add(slot_var["AI"] != 1).OnlyEnforceIf(ai_not_slot1)
    model.Add(slot_var["AI"] == 1).OnlyEnforceIf(ai_not_slot1.Not())
    model.Minimize(ai_not_slot1)

    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("=== Exam Timetable ===")
        for c in courses:
            print(f"  {c:5s} → Slot {solver.Value(slot_var[c])}, "
                  f"Room R{solver.Value(room_var[c])}")
        print(f"AI preference violations: {solver.ObjectiveValue()}")
    else:
        print("No solution found.")

# csp_exam_timetable()

# ══ TEMPLATE B: N-Queens ══
def csp_nqueens(n=8):
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # queens[i] = row of queen in column i
    queens = [model.NewIntVar(0, n-1, f"q{i}") for i in range(n)]

    # All different rows
    model.AddAllDifferent(queens)

    # All different diagonals
    diag1 = [queens[i] - i for i in range(n)]
    diag2 = [queens[i] + i for i in range(n)]
    model.AddAllDifferent(diag1)
    model.AddAllDifferent(diag2)

    solutions = []
    class SolutionPrinter(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self._count = 0
        def on_solution_callback(self):
            self._count += 1
            sol = [self.Value(queens[i]) for i in range(n)]
            solutions.append(sol)
            if self._count <= 3:          # print first 3
                print(f"\nSolution {self._count}:")
                for row in range(n):
                    line = ""
                    for col in range(n):
                        line += "Q " if sol[col] == row else ". "
                    print(line)

    cb = SolutionPrinter()
    solver.parameters.enumerate_all_solutions = True
    solver.Solve(model, cb)
    print(f"\nTotal solutions for {n}-Queens: {cb._count}")

# csp_nqueens(8)

# ══ TEMPLATE C: Resource Allocation / LP ══
def csp_resource_allocation():
    """
    Maximize: 7F + 6W + 9M
    Subject to linear constraints.
    CHANGE: coefficients and limits below.
    """
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # Decision variables (non-negative integers)
    F = model.NewIntVar(0, 100, "F")   # food kits
    W = model.NewIntVar(0, 100, "W")   # water packs
    M = model.NewIntVar(0, 100, "M")   # medicine

    # Constraints — CHANGE coefficients and limits
    model.Add(4*F + 3*W + 5*M <= 180)  # volume
    model.Add(6*F + 2*W + 4*M <= 200)  # weight
    model.Add(3*F + 5*W + 2*M <= 150)  # budget
    model.Add(W >= 10)                  # minimum water
    model.Add(M >= 8)                   # minimum medicine

    # Objective: maximise impact
    model.Maximize(7*F + 6*W + 9*M)

    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        print("=== Resource Allocation ===")
        print(f"  F (food)     = {solver.Value(F)}")
        print(f"  W (water)    = {solver.Value(W)}")
        print(f"  M (medicine) = {solver.Value(M)}")
        print(f"  Impact Score = {solver.ObjectiveValue()}")
        print(f"  Conflicts    = {solver.NumConflicts()}")
        print(f"  Branches     = {solver.NumBranches()}")
        print(f"  Wall time    = {solver.WallTime():.4f}s")

    # What-if A: budget tightens to 130
    model2  = cp_model.CpModel()
    F2 = model2.NewIntVar(0,100,"F"); W2=model2.NewIntVar(0,100,"W"); M2=model2.NewIntVar(0,100,"M")
    model2.Add(4*F2+3*W2+5*M2<=180); model2.Add(6*F2+2*W2+4*M2<=200)
    model2.Add(3*F2+5*W2+2*M2<=130); model2.Add(W2>=10); model2.Add(M2>=8)
    model2.Maximize(7*F2+6*W2+9*M2)
    s2 = cp_model.CpSolver(); s2.Solve(model2)
    print(f"\nCase A (budget=130): Impact={s2.ObjectiveValue()}, F={s2.Value(F2)}, W={s2.Value(W2)}, M={s2.Value(M2)}")

    # What-if B: min medicine = 15
    model3  = cp_model.CpModel()
    F3=model3.NewIntVar(0,100,"F"); W3=model3.NewIntVar(0,100,"W"); M3=model3.NewIntVar(0,100,"M")
    model3.Add(4*F3+3*W3+5*M3<=180); model3.Add(6*F3+2*W3+4*M3<=200)
    model3.Add(3*F3+5*W3+2*M3<=150); model3.Add(W3>=10); model3.Add(M3>=15)
    model3.Maximize(7*F3+6*W3+9*M3)
    s3 = cp_model.CpSolver(); s3.Solve(model3)
    print(f"Case B (minM=15):   Impact={s3.ObjectiveValue()}, F={s3.Value(F3)}, W={s3.Value(W3)}, M={s3.Value(M3)}")

# csp_resource_allocation()

# ══ TEMPLATE D: Traffic Light / Arc Consistency ══
def csp_traffic():
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()

    # 0=NS, 1=EW
    A = model.NewIntVar(0,1,"A"); B=model.NewIntVar(0,1,"B")
    C = model.NewIntVar(0,1,"C"); D=model.NewIntVar(0,1,"D")

    # Unary: D cannot use EW (1)
    model.Add(D != 1)

    # Binary: adjacent != same
    adjacency = [(A,B),(A,C),(B,D),(C,D)]
    for (x,y) in adjacency:
        model.Add(x != y)

    solutions = []
    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            solutions.append({
                "A":self.Value(A),"B":self.Value(B),
                "C":self.Value(C),"D":self.Value(D)
            })
    cb = CB()
    solver.parameters.enumerate_all_solutions = True
    solver.Solve(model, cb)
    labels = {0:"NS", 1:"EW"}
    print("=== Traffic Light Solutions ===")
    for sol in solutions:
        print({k: labels[v] for k,v in sol.items()})

# csp_traffic()


# ═══════════════════════════════════════════════════════════════════
# [7]  EDA  (Exploratory Data Analysis)
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "dataset", "explore", "visualize", "distribution",
                   "correlation", "missing values", "summary"
CHANGE: filepath or use the synthetic generator below.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_template(df=None, filepath=None):
    """Pass either a DataFrame or a CSV filepath."""
    if df is None and filepath:
        df = pd.read_csv(filepath)
    elif df is None:
        # ── Synthetic data for demo ──
        np.random.seed(42)
        df = pd.DataFrame({
            "age":    np.random.randint(18,65,100),
            "salary": np.random.randint(30000,120000,100),
            "score":  np.random.rand(100)*100,
            "dept":   np.random.choice(["IT","HR","Finance"],100)
        })

    print("=== SHAPE ==="); print(df.shape)
    print("\n=== HEAD ===");  print(df.head())
    print("\n=== INFO ===");  print(df.info())
    print("\n=== DESCRIBE ==="); print(df.describe())
    print("\n=== MISSING VALUES ==="); print(df.isnull().sum())
    print("\n=== CORRELATION ===");
    print(df.select_dtypes(include=[np.number]).corr())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histograms
    for i, col in enumerate(numeric_cols[:2]):
        df[col].hist(ax=axes[0,i], bins=20, color="steelblue", edgecolor="black")
        axes[0,i].set_title(f"Distribution: {col}")

    # Correlation heatmap
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, ax=axes[1,0], annot=True, cmap="coolwarm", fmt=".2f")
    axes[1,0].set_title("Correlation Heatmap")

    # Boxplot
    if len(numeric_cols) >= 1:
        df.boxplot(column=numeric_cols[0], ax=axes[1,1])
        axes[1,1].set_title(f"Boxplot: {numeric_cols[0]}")

    plt.tight_layout()
    plt.savefig("eda_output.png"); plt.show()
    print("EDA complete. Plot saved as eda_output.png")

# eda_template()


# ═══════════════════════════════════════════════════════════════════
# [8]  ML ALGORITHMS  (Supervised + Unsupervised)
# ═══════════════════════════════════════════════════════════════════
"""
SCENARIO KEYWORDS: "classify", "predict", "cluster", "train", "test",
                   "accuracy", "decision tree", "KNN", "SVM", "KMeans",
                   "regression", "logistic", "random forest"
CHANGE: dataset (use any sklearn dataset or CSV), model choice.
"""

from sklearn.datasets      import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, classification_report, confusion_matrix
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.cluster       import KMeans
import warnings
warnings.filterwarnings("ignore")

# ══ Part A: Supervised Classification ══
def supervised_ml(dataset="iris"):
    # Load data — CHANGE to your scenario
    if dataset == "iris":
        data = load_iris()
    elif dataset == "cancer":
        data = load_breast_cancer()
    else:
        data = load_iris()  # default fallback

    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── All models ──
    models = {
        "Decision Tree"    : DecisionTreeClassifier(max_depth=5, random_state=42),
        "KNN (k=5)"        : KNeighborsClassifier(n_neighbors=5),
        "SVM"              : SVC(kernel="rbf", random_state=42),
        "Random Forest"    : RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Reg."    : LogisticRegression(max_iter=200, random_state=42),
    }

    best_acc, best_name = 0, ""
    print(f"=== Supervised Learning on {dataset.upper()} ===\n")
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc  = accuracy_score(y_test, pred)
        print(f"{name:20s}: Accuracy = {acc:.4f}")
        if acc > best_acc: best_acc, best_name = acc, name

    print(f"\nBest Model: {best_name} ({best_acc:.4f})")

    # Detailed report for best model
    best_clf = models[best_name]
    pred = best_clf.predict(X_test)
    print(f"\nClassification Report for {best_name}:")
    print(classification_report(y_test, pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    return models

# supervised_ml("iris")

# ══ Part B: Unsupervised — K-Means Clustering ══
def unsupervised_ml(n_clusters=3):
    from sklearn.decomposition import PCA

    data = load_iris()
    X    = data.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow method to find best k
    inertias = []
    k_range  = range(2, 9)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6,4))
    plt.plot(list(k_range), inertias, "bo-")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("Elbow Method — Optimal k")
    plt.tight_layout(); plt.savefig("elbow.png"); plt.show()

    # Cluster with chosen k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 2D PCA for visualisation
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6,4))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="viridis", s=40)
    plt.title(f"KMeans Clusters (k={n_clusters})")
    plt.tight_layout(); plt.savefig("clusters.png"); plt.show()

    print(f"KMeans with k={n_clusters} — Inertia: {kmeans.inertia_:.2f}")
    print(f"Cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")

# unsupervised_ml(3)


# ═══════════════════════════════════════════════════════════════════
# QUICK REFERENCE — what to change for each exam question
# ═══════════════════════════════════════════════════════════════════
"""
┌─────────────────────────┬────────────────────────────────────────┐
│ QUESTION TYPE           │ WHAT TO CHANGE                         │
├─────────────────────────┼────────────────────────────────────────┤
│ Tic-Tac-Toe / board     │ Nothing — copy Section [1] as-is       │
│ Coin row / game tree    │ Section [2]: change coins list          │
│ Cyber / complex game    │ Section [3]: change heuristic formula   │
│                         │   and actions list                     │
│ Bayesian disease/exam   │ Section [4]: change edges, CPD values   │
│                         │   and inference query                  │
│ Markov / weather        │ Section [5]: change states, matrix,     │
│                         │   initial_state, num_steps             │
│ Timetable / scheduling  │ Section [6A]: change courses, conflicts │
│ N-Queens                │ Section [6B]: change n                  │
│ Resource allocation     │ Section [6C]: change coefficients       │
│ Traffic lights          │ Section [6D]: change adjacency pairs    │
│ EDA                     │ Section [7]: change filepath or df      │
│ Classification          │ Section [8A]: change dataset & model    │
│ Clustering              │ Section [8B]: change n_clusters         │
└─────────────────────────┴────────────────────────────────────────┘

ENCODING REMINDER for pgmpy:
  - variable_card = number of states (2 for binary, 3 for A/B/C)
  - values rows = [P(state0|...), P(state1|...), ...]
  - values cols = combinations of parent states (left-to-right)
  - evidence=1 usually means YES/True/High (second state in values)
  - evidence=0 usually means NO/False/Low  (first  state in values)

INSTALL COMMANDS (if needed):
  pip install pgmpy ortools scikit-learn pandas seaborn matplotlib numpy
"""

# ─── MAIN: uncomment the function you need ───
if __name__ == "__main__":
    pass
    # play_tictactoe()
    # run_coin_row([6, 2, 9, 4, 7, 3, 8, 5])
    # simulate_game({"health":80,"vuln":5,"intrusion":3,"resources":10})
    # bayesian_disease()
    # bayesian_student()
    # bayesian_burglary()
    # markov_weather()
    # csp_exam_timetable()
    # csp_nqueens(8)
    # csp_resource_allocation()
    # csp_traffic()
    # eda_template()
    # supervised_ml("iris")
    # unsupervised_ml(3)
